#include "calibration/calib.h"

// std
#include <numeric>

// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "calibration/distortion.h"
#include "calibration/planarpose.h"  // for estimate_planar_pose_dlt

#include "observationutils.h"

namespace vitavision {

using Pose6 = Eigen::Matrix<double, 6, 1>;

// Variable projection residual for full camera calibration.
struct CalibVPResidual {
    using ObsBuffer = std::vector<Observation<double>>;

    std::vector<std::vector<PlanarObservation>> views_;  // observations per view
    int num_radial_;
    size_t total_obs_;
    mutable ObsBuffer obs_;

    CalibVPResidual(std::vector<std::vector<PlanarObservation>> views, int num_radial)
        : views_(std::move(views)), num_radial_(num_radial) {
        total_obs_ = 0;
        for (const auto& v : views_) total_obs_ += v.size();
        obs_.resize(total_obs_);
    }

    template<typename T>
    bool operator()(T const* const* params, T* residuals) const {
        const T* intr = params[0];
        if (obs_.size() != total_obs_) obs_.resize(total_obs_);

        size_t obs_idx = 0;
        for (size_t i = 0; i < views_.size(); ++i) {
            const T* pose6 = params[1 + i];
            for (const auto& ob : views_[i]) {
                obs_[obs_idx++] = to_observation(ob, pose6);
            }
        }
        auto [_, r] = fit_distortion_full(obs_, intr[0], intr[1], intr[2], intr[3], num_radial_);
        for (int i = 0; i < r.size(); ++i) residuals[i] = r[i];
        return true;
    }

    DistortionWithResiduals<double> solve_full(const double* intr, const std::vector<const double*>& pose_ptrs) const {
        if (obs_.size() != total_obs_) obs_.resize(total_obs_);

        size_t obs_idx = 0;
        for (size_t i = 0; i < views_.size(); ++i) {
            const double* pose6 = pose_ptrs[i];
            for (const auto& ob : views_[i]) {
                obs_[obs_idx++] = to_observation(ob, pose6);
            }
        }
        return fit_distortion_full(obs_, intr[0], intr[1], intr[2], intr[3], num_radial_);
    }
};

static Eigen::Affine3d axisangle_to_pose(const Pose6& pose6) {
    Eigen::Matrix3d R;
    ceres::AngleAxisToRotationMatrix(pose6.head<3>().data(), R.data());
    Eigen::Affine3d T = Eigen::Affine3d::Identity();
    T.linear() = R;
    T.translation() = pose6.tail<3>();
    return T;
}

static std::pair<size_t, std::vector<std::vector<PlanarObservation>>> prepare_observations(const std::vector<PlanarView>& views) {
    std::vector<std::vector<PlanarObservation>> obs_views;
    size_t total_obs = 0;
    obs_views.reserve(views.size());

    size_t i = 0;
    for (const auto& view : views) {
        ++i;
        std::vector<PlanarObservation> view_observations;
        const auto& obj = view.object_xy;
        const auto& img = view.image_uv;

        if (obj.size() != img.size()) {
            std::cerr << "Object and image coordinates must have the same size in view " << i - 1 << "." << std::endl;
            continue;
        }

        const size_t n = obj.size();
        if (n < 8) {
            std::cerr << "Insufficient points for calibration in view " << i - 1 << "." << std::endl;
            continue;
        }

        view_observations.reserve(n);
        for (size_t j = 0; j < n; ++j) {
            view_observations.push_back({obj[j], img[j]});
        }
        total_obs += n;
        obs_views.emplace_back(std::move(view_observations));
    }

    return {total_obs, obs_views};
}

// Initialize camera poses from views
static void initialize_poses(
    const std::vector<PlanarView>& views,
    const CameraMatrix& initial_guess,
    std::vector<Pose6>& poses
) {
    for (size_t i = 0; i < views.size(); ++i) {
        Eigen::Affine3d pose = estimate_planar_pose_dlt(views[i].object_xy, views[i].image_uv, initial_guess);
        ceres::RotationMatrixToAngleAxis(pose.linear().data(), poses[i].data());
        poses[i].tail<3>() = pose.translation();
    }
}

// Set up the Ceres optimization problem
static void setup_optimization_problem(
    const std::vector<std::vector<PlanarObservation>>& obs_views,
    size_t total_obs,
    int num_radial,
    double* intrinsics,
    std::vector<Pose6>& poses,
    ceres::Problem& problem
) {
    auto* functor = new CalibVPResidual(obs_views, num_radial);
    auto* cost = new ceres::DynamicAutoDiffCostFunction<CalibVPResidual>(functor);
    cost->AddParameterBlock(4);  // Intrinsics
    for (size_t i = 0; i < poses.size(); ++i) {
        cost->AddParameterBlock(6);  // Pose for each view
    }
    cost->SetNumResiduals(static_cast<int>(total_obs * 2));

    // Add parameter blocks to the problem
    std::vector<double*> param_blocks;
    param_blocks.push_back(intrinsics);
    for (size_t i = 0; i < poses.size(); ++i) {
        param_blocks.push_back(poses[i].data());
    }
    problem.AddResidualBlock(cost, nullptr, param_blocks);
}

// Calculate reprojection errors overall and per view
static void compute_reprojection_errors(
    const std::vector<std::vector<PlanarObservation>>& obs_views,
    size_t total_obs,
    const Eigen::VectorXd& residuals,
    CameraCalibrationResult& result
) {
    int total_residuals = static_cast<int>(total_obs * 2);
    double sum_squared_residuals = residuals.squaredNorm();
    result.reprojection_error = std::sqrt(sum_squared_residuals / total_residuals);
    
    const size_t num_views = obs_views.size();
    result.view_errors.resize(num_views);
    int residual_idx = 0;
    
    for (size_t i = 0; i < num_views; ++i) {
        int view_points = static_cast<int>(obs_views[i].size());
        double view_error_sum = 0.0;
        for (int j = 0; j < view_points * 2; ++j) {
            double r = residuals[residual_idx++];
            view_error_sum += r * r;
        }
        result.view_errors[i] = std::sqrt(view_error_sum / (view_points * 2));
    }
}

// Populate the result with intrinsics and poses
static void populate_result_parameters(
    const double* intrinsics,
    const std::vector<Pose6>& poses,
    CameraCalibrationResult& result
) {
    result.intrinsics.fx = intrinsics[0];
    result.intrinsics.fy = intrinsics[1];
    result.intrinsics.cx = intrinsics[2];
    result.intrinsics.cy = intrinsics[3];
    
    result.poses.resize(poses.size());
    for (size_t i = 0; i < poses.size(); ++i) {
        result.poses[i] = axisangle_to_pose(poses[i]);
    }
}

// Compute and populate the covariance matrix
static bool compute_covariance(
    const std::vector<double*>& param_blocks,
    const std::vector<int>& block_sizes,
    size_t total_params,
    size_t total_residuals,
    double sum_squared_residuals,
    ceres::Problem& problem,
    CameraCalibrationResult& result
) {
    // Convert param_blocks to const pointers
    std::vector<const double*> const_param_blocks;
    for (const auto* ptr : param_blocks) {
        const_param_blocks.push_back(ptr);
    }
    
    ceres::Covariance::Options cov_options;
    ceres::Covariance covariance(cov_options);
    std::vector<std::pair<const double*, const double*>> cov_blocks;
    
    for (size_t i = 0; i < const_param_blocks.size(); ++i) {
        for (size_t j = 0; j <= i; ++j) {
            cov_blocks.emplace_back(const_param_blocks[i], const_param_blocks[j]);
        }
    }

    if (!covariance.Compute(cov_blocks, &problem)) {
        return false;
    }

    Eigen::MatrixXd cov_matrix = Eigen::MatrixXd::Zero(total_params, total_params);
    size_t row_offset = 0;
    
    for (size_t i = 0; i < const_param_blocks.size(); ++i) {
        int block_i_size = block_sizes[i];
        size_t col_offset = 0;
        
        for (size_t j = 0; j <= i; ++j) {
            int block_j_size = block_sizes[j];
            std::vector<double> block_cov(block_i_size * block_j_size);
            
            covariance.GetCovarianceBlock(const_param_blocks[i], const_param_blocks[j], block_cov.data());
            
            for (int r = 0; r < block_i_size; ++r) {
                for (int c = 0; c < block_j_size; ++c) {
                    double value = block_cov[r * block_j_size + c];
                    cov_matrix(row_offset + r, col_offset + c) = value;
                    if (j < i) {
                        cov_matrix(col_offset + c, row_offset + r) = value;
                    }
                }
            }
            col_offset += block_j_size;
        }
        row_offset += block_i_size;
    }

    // Scale covariance by variance factor
    int degrees_of_freedom = std::max(1, static_cast<int>(total_residuals) - static_cast<int>(total_params));
    double variance_factor = sum_squared_residuals / degrees_of_freedom;
    cov_matrix *= variance_factor;

    result.covariance = cov_matrix;
    return true;
}

CameraCalibrationResult calibrate_camera_planar(
    const std::vector<PlanarView>& views,
    int num_radial,
    const CameraMatrix& initial_guess,
    bool verbose
) {
    CameraCalibrationResult result;

    // Prepare observations per view
    auto [total_obs, obs_views] = prepare_observations(views);
    const size_t num_views = obs_views.size();
    if (num_views < 4) {
        std::cerr << "Insufficient views for calibration (at least 4 required)." << std::endl;
        return result;
    }

    // Initialize intrinsics and poses
    double intrinsics[4] = {initial_guess.fx, initial_guess.fy, initial_guess.cx, initial_guess.cy};
    std::vector<Pose6> poses(num_views);
    initialize_poses(views, initial_guess, poses);

    // Set up and solve the optimization problem
    ceres::Problem problem;
    setup_optimization_problem(obs_views, total_obs, num_radial, intrinsics, poses, problem);

    // Collect parameter blocks for later use
    std::vector<double*> param_blocks;
    param_blocks.push_back(intrinsics);
    for (size_t i = 0; i < num_views; ++i) {
        param_blocks.push_back(poses[i].data());
    }

    // Configure and run the solver
    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.minimizer_progress_to_stdout = verbose;

    constexpr double eps = 1e-6;
    opts.function_tolerance = eps;
    opts.gradient_tolerance = eps;
    opts.parameter_tolerance = eps;
    opts.max_num_iterations = 1000;

    #if 1
    // TODO: make it dependent on the image size
    // Add parameter bounds to prevent divergence
    problem.SetParameterLowerBound(intrinsics, 0, 100.0);  // fx > 100
    problem.SetParameterLowerBound(intrinsics, 1, 100.0);  // fy > 100
    problem.SetParameterLowerBound(intrinsics, 2, 10.0);   // cx > 10
    problem.SetParameterLowerBound(intrinsics, 3, 10.0);   // cy > 10

    problem.SetParameterUpperBound(intrinsics, 0, 2000.0); // fx < 2000
    problem.SetParameterUpperBound(intrinsics, 1, 2000.0); // fy < 2000
    problem.SetParameterUpperBound(intrinsics, 2, 1280.0); // cx < 1280
    problem.SetParameterUpperBound(intrinsics, 3, 720.0);  // cy < 720
    #endif

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);
    result.summary = summary.BriefReport();

    // Compute best distortion and residuals
    std::vector<const double*> pose_ptrs(num_views);
    for (size_t i = 0; i < num_views; ++i) {
        pose_ptrs[i] = poses[i].data();
    }

    CalibVPResidual functor(obs_views, num_radial);
    auto dr = functor.solve_full(intrinsics, pose_ptrs);
    result.distortion = dr.distortion;

    // Process results
    compute_reprojection_errors(obs_views, total_obs, dr.residuals, result);
    populate_result_parameters(intrinsics, poses, result);

    // Compute covariance matrix
    const size_t total_params = 4 + 6 * num_views;
    std::vector<int> block_sizes;
    block_sizes.push_back(4);  // Intrinsics block
    for (size_t i = 0; i < num_views; ++i) {
        block_sizes.push_back(6);  // Pose blocks
    }

    double sum_squared_residuals = dr.residuals.squaredNorm();
    size_t total_residuals = total_obs * 2;
    compute_covariance(param_blocks, block_sizes, total_params, total_residuals,
                       sum_squared_residuals, problem, result);

    return result;
}

} // namespace vitavision

