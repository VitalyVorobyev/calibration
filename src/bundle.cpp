#include "calibration/bundle.h"

// std
#include <numeric>
#include <array>

// eigen
#include <Eigen/Geometry>

// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/manifold.h>

#include "calibration/planarpose.h"

#include "observationutils.h"
#include "bundleresidual.h"

constexpr double CONVERGENCE_TOLERANCE = 1e-6;
constexpr int MAX_ITERATIONS = 1000;

namespace vitavision {

struct BundleParamBlocks final {
    std::array<double, 4> b_q_t{};             // target to base
    std::array<double, 3> b_t_t{};
    std::array<double, 4> g_q_r{};             // reference to gripper (hand-eye pose)
    std::array<double, 3> g_t_r{};
    std::vector<std::array<double, 4>> c_q_r;  // reference to camera (extrinsic poses)
    std::vector<std::array<double, 3>> c_t_r;
    std::vector<std::array<double, 9>> intr;   // camera matrices and distortions
};

static BundleParamBlocks initialize_blocks(
    const std::vector<Camera>& initial_cameras,
    const Eigen::Affine3d& g_T_r,
    const std::vector<Eigen::Affine3d>& c_T_r,
    const Eigen::Affine3d& b_T_t
) {
    BundleParamBlocks blocks;
    const size_t ncam = c_T_r.size();

    populate_quat_tran(g_T_r, blocks.g_q_r, blocks.g_t_r);
    populate_quat_tran(b_T_t, blocks.b_q_t, blocks.b_t_t);

    blocks.c_q_r.resize(ncam);
    blocks.c_t_r.resize(ncam);
    blocks.intr.resize(ncam);
    for (size_t i = 0; i < c_T_r.size(); ++i) {
        populate_quat_tran(c_T_r[i], blocks.c_q_r[i], blocks.c_t_r[i]);

        const auto& cam = initial_cameras[i];
        std::array<double,9> intr = {
            cam.K.fx,
            cam.K.fy,
            cam.K.cx,
            cam.K.cy,
            0,0,0,0,0
        };
        for (int d = 0; d < std::min<int>(5, cam.distortion.forward.size()); ++d) {
            intr[4 + d] = cam.distortion.forward(d);
        }
        blocks.intr[i] = intr;
    }

    return blocks;
}

static void build_problem(const std::vector<BundleObservation>& observations,
                          const BundleOptions& opts,
                          BundleParamBlocks& blocks,
                          ceres::Problem& p) {
    for (const auto& obs : observations) {
        const size_t cam = obs.camera_index;
        auto* cost = HandEyeReprojResidual::create(obs.view, obs.b_T_g);
        p.AddResidualBlock(cost, new ceres::HuberLoss(1.0),
                           blocks.b_q_t.data(), blocks.b_t_t.data(),
                           blocks.g_q_r.data(), blocks.g_t_r.data(),
                           blocks.c_q_r[cam].data(), blocks.c_t_r[cam].data(),
                           blocks.intr[cam].data());
    }

    // keep identity extrinsic constant
    p.SetParameterBlockConstant(blocks.c_q_r[0].data());
    p.SetParameterBlockConstant(blocks.c_t_r[0].data());
    const bool single_cam = blocks.intr.size() == 1;

    // set quaternions
    p.SetManifold(blocks.b_q_t.data(), new ceres::QuaternionManifold());
    p.SetManifold(blocks.g_q_r.data(), new ceres::QuaternionManifold());
    for (size_t cam = 0; cam < blocks.c_q_r.size(); ++cam) {
        p.SetManifold(blocks.c_q_r[cam].data(), new ceres::QuaternionManifold());
    }

    // With a single camera, the hand-eye and target pose cannot be estimated
    // simultaneously. If both are requested, fix the target pose to the
    // initial guess to remove the gauge freedom.
    if (!opts.optimize_target_pose || (single_cam && opts.optimize_hand_eye)) {
        p.SetParameterBlockConstant(blocks.b_q_t.data());
        p.SetParameterBlockConstant(blocks.b_t_t.data());
    }

    if (!opts.optimize_hand_eye) {
        p.SetParameterBlockConstant(blocks.g_q_r.data());
        p.SetParameterBlockConstant(blocks.g_t_r.data());
    }

    if (!opts.optimize_extrinsics) {
        for (auto& e : blocks.c_q_r) p.SetParameterBlockConstant(e.data());
        for (auto& e : blocks.c_t_r) p.SetParameterBlockConstant(e.data());
    }

    if (!opts.optimize_intrinsics) {
        for (size_t c = 0; c < blocks.intr.size(); ++c) {
            p.SetParameterBlockConstant(blocks.intr[c].data());
        }
    } else {
        // ensure fx > 0 and fy > 0
        for (size_t c = 0; c < blocks.intr.size(); ++c) {
            p.SetParameterLowerBound(blocks.intr[c].data(), 0, 0.0);
            p.SetParameterLowerBound(blocks.intr[c].data(), 1, 0.0);
        }
        // Anchor scale by fixing fx, fy for reference camera
        p.SetManifold(blocks.intr[0].data(), new ceres::SubsetManifold(4, {0, 1}));
    }
}

static void recover_parameters(
    const BundleParamBlocks& blocks,
    BundleResult& result
) {
    result.b_T_t = restore_pose(blocks.b_q_t, blocks.b_t_t);
    result.g_T_r = restore_pose(blocks.g_q_r, blocks.g_t_r);

    const size_t num_cams = blocks.intr.size();
    result.c_T_r.resize(num_cams);
    result.cameras.resize(num_cams);

    for (size_t c = 0; c < num_cams; ++c) {
        result.c_T_r[c] = restore_pose(blocks.c_q_r[c], blocks.c_t_r[c]);
        const auto& i = blocks.intr[c];
        CameraMatrix K{i[0], i[1], i[2], i[3]};
        Eigen::VectorXd dist(5);
        dist << i[4], i[5], i[6], i[7], i[8];
        result.cameras[c] = Camera(K, dist);
    }
}

static double compute_reprojection_error(
    const std::vector<BundleObservation>& observations,
    BundleResult& result
) {
    double ssr = 0.0; size_t total = 0;

    for (const auto& obs : observations) {
        const size_t cam_idx = obs.camera_index;
        const auto& cam = result.cameras[cam_idx];
        const auto& intr = cam.K;
        const Eigen::VectorXd& dist = cam.distortion.forward;

        auto c_T_t = get_camera_T_target(
            result.b_T_t, result.g_T_r, result.c_T_r[cam_idx], obs.b_T_g);

        double u_hat, v_hat;
        std::array<double, 9> i {
            intr.fx, intr.fy, intr.cx, intr.cy,
            dist.size() > 0 ? dist(0) : 0.0,
            dist.size() > 1 ? dist(1) : 0.0,
            dist.size() > 2 ? dist(2) : 0.0,
            dist.size() > 3 ? dist(3) : 0.0,
            dist.size() > 4 ? dist(4) : 0.0
        };
        Eigen::Vector3d P;
        for (const auto& ob : obs.view) {
            P << ob.object_xy.x(), ob.object_xy.y(), 0;
            P = c_T_t * P;
            project_with_intrinsics(P(0), P(1), P(2), i.data(), true, u_hat, v_hat);
            double du = u_hat - ob.image_uv.x();
            double dv = v_hat - ob.image_uv.y();
            ssr += du * du + dv * dv;
            total += 2;
        }
    }

    return total ? std::sqrt(ssr / total) : 0.0;
}

static Eigen::MatrixXd compute_covariance(ceres::Problem& p,
                                          BundleParamBlocks& blocks) {
    std::vector<const double*> blocks_list;
    blocks_list.push_back(blocks.b_q_t.data());
    blocks_list.push_back(blocks.b_t_t.data());
    blocks_list.push_back(blocks.g_q_r.data());
    blocks_list.push_back(blocks.g_t_r.data());
    for (auto& e : blocks.c_q_r) blocks_list.push_back(e.data());
    for (auto& e : blocks.c_t_r) blocks_list.push_back(e.data());
    for (auto& e : blocks.intr) blocks_list.push_back(e.data());

    ceres::Covariance::Options cov_opts;
    ceres::Covariance cov(cov_opts);
    if (!cov.Compute(blocks_list, &p)) return Eigen::MatrixXd();

    std::vector<int> sizes(blocks_list.size());
    for (size_t i = 0; i < blocks_list.size(); ++i) {
        sizes[i] = p.ParameterBlockSize(blocks_list[i]);
    }
    int dim = 0; for (int s : sizes) dim += s;

    Eigen::MatrixXd cov_mat = Eigen::MatrixXd::Zero(dim, dim);
    cov.GetCovarianceMatrix(blocks_list, cov_mat.data());
    return cov_mat;
}

static std::string solve_problem(ceres::Problem &p, bool verbose) {
    ceres::Solver::Options sopts;
    #if 0
    sopts.linear_solver_type = ceres::SPARSE_SCHUR;
    //  ;// ceres::DENSE_QR;
    #elif 0
    sopts.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;  // default option
    #elif 0
    sopts.linear_solver_type = ceres::DENSE_SCHUR;
    #endif
    sopts.minimizer_progress_to_stdout = verbose;

    constexpr double eps = CONVERGENCE_TOLERANCE;
    sopts.function_tolerance = eps;
    sopts.gradient_tolerance = eps;
    sopts.parameter_tolerance = eps;
    sopts.max_num_iterations = MAX_ITERATIONS;

    ceres::Solver::Summary summary;
    ceres::Solve(sopts, &p, &summary);
    return summary.BriefReport();
}

BundleResult optimize_bundle(
    const std::vector<BundleObservation>& observations,
    const std::vector<Camera>& initial_cameras,
    const Eigen::Affine3d& init_g_T_r,
    const std::vector<Eigen::Affine3d>& init_c_T_r,
    const Eigen::Affine3d& init_b_T_t,
    const BundleOptions& opts
) {
    const size_t num_cams = initial_cameras.size();
    if (num_cams == 0) {
        throw std::invalid_argument("No camera intrinsics provided");
    };

    BundleParamBlocks blocks = initialize_blocks(
        initial_cameras, init_g_T_r,
        init_c_T_r, init_b_T_t);

    ceres::Problem p;
    build_problem(observations, opts, blocks, p);

    BundleResult result;
    result.report = solve_problem(p, opts.verbose);

    recover_parameters(blocks, result);
    result.reprojection_error = compute_reprojection_error(observations, result);
    result.covariance = compute_covariance(p, blocks);

    return result;
}

}  // namespace vitavision
