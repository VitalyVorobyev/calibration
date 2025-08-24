#include "calibration/handeye.h"

// std
#include <numeric>
#include <array>

// eigen
#include <Eigen/Geometry>

// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/manifold.h>
#include <ceres/sphere_manifold.h>

#include "calibration/planarpose.h"

#include "observationutils.h"

constexpr int DEFAULT_NUM_RADIAL_DISTORTION_PARAMS = 2;
constexpr double CONVERGENCE_TOLERANCE = 1e-6;
constexpr int MAX_ITERATIONS = 1000;

namespace vitavision {

static Eigen::Matrix3d estimate_hand_eye_rotation(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& camera_T_target
) {
    if (base_T_gripper.empty() || base_T_gripper.size() != camera_T_target.size()) {
        std::cerr << "Inconsistent number of poses\n";
        throw std::runtime_error("Inconsistent number of poses");
    }
    const size_t m = base_T_gripper.size() - 1;  // number of motion pairs
    Eigen::MatrixXd M(3*m, 3);
    Eigen::VectorXd d(3*m);

    for (size_t i = 0; i < m; ++i) {
        auto alpha = log_rot(base_T_gripper[i].linear().transpose() * base_T_gripper[i+1].linear());
        auto beta = log_rot(camera_T_target[i].linear() * camera_T_target[i+1].linear().transpose());
        M.block<3,3>(3*i,0) = skew(alpha + beta);
        d.segment<3>(3*i) = beta - alpha;
    }

    Eigen::Vector3d r = solve_llsq(M, d);
    double angle = r.norm();
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    if (angle > 1e-12) {
        R = Eigen::AngleAxisd(angle, r.normalized()).toRotationMatrix();
    }
    return R;
}

static Eigen::Vector3d estimate_hand_eye_translation(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& camera_T_target,
    const Eigen::Matrix3d& R
) {
    if (base_T_gripper.empty() || base_T_gripper.size() != camera_T_target.size()) {
        std::cerr << "Inconsistent number of poses\n";
        throw std::runtime_error("Inconsistent number of poses");
    }
    const size_t m = base_T_gripper.size() - 1;  // number of motion pairs

    Eigen::MatrixXd C(3*m, 3);
    Eigen::VectorXd w(3*m);
    for (size_t i = 0; i < m; ++i) {
        Eigen::Affine3d A = base_T_gripper[i].inverse() * base_T_gripper[i+1];
        Eigen::Affine3d B = camera_T_target[i] * camera_T_target[i+1].inverse();
        C.block<3,3>(3*i,0) = A.linear() - Eigen::Matrix3d::Identity();
        w.segment<3>(3*i) = R * B.translation() - A.translation();
    }
    Eigen::Vector3d t = solve_llsq(C, w);
    return t;
}

Eigen::Affine3d estimate_hand_eye_initial(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& camera_T_target) {

    const size_t n = base_T_gripper.size();
    if (n < 2 || n != camera_T_target.size()) {
        std::cerr << "Insufficient data for initial hand-eye estimate\n";
        return Eigen::Affine3d::Identity();
    }
    auto R = estimate_hand_eye_rotation(base_T_gripper, camera_T_target);
    auto t = estimate_hand_eye_translation(base_T_gripper, camera_T_target, R);

    Eigen::Affine3d X = Eigen::Affine3d::Identity();
    X.linear() = R;
    X.translation() = t;
    return X;
}

// Comnutes target -> camera transform
template<typename T>
static Eigen::Transform<T, 3, Eigen::Affine> get_camera_T_target(
    const Eigen::Transform<T, 3, Eigen::Affine>& base_T_target,
    const Eigen::Transform<T, 3, Eigen::Affine>& refcam_T_gripper,
    const Eigen::Transform<T, 3, Eigen::Affine>& camera_T_refcam,
    const Eigen::Transform<T, 3, Eigen::Affine>& base_T_gripper
) {
    auto camera_T_gripper = camera_T_refcam * refcam_T_gripper;  // gripper -> camera
    auto camera_T_base = camera_T_gripper * base_T_gripper.inverse();  // base -> camera
    auto camera_T_target = camera_T_base * base_T_target;  // target -> camera
    return camera_T_target;
}

struct HandEyeReprojResidual final {
    PlanarView view;
    Eigen::Affine3d base_to_gripper;
    HandEyeReprojResidual(PlanarView v, const Eigen::Affine3d& base_T_gripper)
        : view(std::move(v)), base_to_gripper(base_T_gripper) {}

    template <typename T>
    bool operator()(const T* base_target6, const T* he_ref6, const T* ext6,
                    const T* intrinsics, const T* dist, T* residuals) const {
        auto base_T_target = pose2affine(base_target6);  // target -> base
        auto refcam_T_gripper = pose2affine(he_ref6);    // gripper -> reference camera
        auto camera_T_refcam = pose2affine(ext6);        // reference -> camera extrinsic
        auto camera_T_target = get_camera_T_target(
            base_T_target, refcam_T_gripper, camera_T_refcam, base_to_gripper.template cast<T>());

        std::vector<Observation<T>> o(view.size());
        planar_observables_to_observables(view, o, camera_T_target);

        const T fx = intrinsics[0];
        const T fy = intrinsics[1];
        const T cx = intrinsics[2];
        const T cy = intrinsics[3];
        Eigen::Map<const Eigen::Matrix<T,Eigen::Dynamic,1>> d(dist, 4);

        int idx = 0;
        for (const auto& ob : o) {
            Eigen::Matrix<T,2,1> norm_xy(ob.x, ob.y);
            Eigen::Matrix<T,2,1> distorted = apply_distortion<T>(norm_xy, d);
            T u = fx * distorted.x() + cx;
            T v = fy * distorted.y() + cy;
            residuals[idx++] = u - ob.u;
            residuals[idx++] = v - ob.v;
        }
        return true;
    }

    static auto* create(PlanarView v, const Eigen::Affine3d& base_T_gripper) {
        auto functor = new HandEyeReprojResidual(v, base_T_gripper);
        auto* cost = new ceres::AutoDiffCostFunction<
            HandEyeReprojResidual, ceres::DYNAMIC, 6,6,6,4,4>(
                functor, static_cast<int>(v.size()) * 2);
        return cost;
    }
};

struct HEParameterBlocks final {
    std::array<double,6> base_target6{};
    std::array<double,6> he_ref6{};
    std::vector<std::array<double,6>> ext6;
    std::vector<std::array<double,4>> K;
    std::vector<std::array<double,4>> dist;
};

static void populate_heref6_block(HEParameterBlocks& blocks,
                        const Eigen::Affine3d& initial_hand_eye) {
    blocks.he_ref6 = pose_to_array(initial_hand_eye);
}

static void populate_extrinsics_blocks(HEParameterBlocks& blocks,
                           const std::vector<Eigen::Affine3d>& initial_extrinsics) {
    blocks.ext6.resize(initial_extrinsics.size());
    std::transform(initial_extrinsics.begin(), initial_extrinsics.end(), blocks.ext6.begin(),
                   [](const Eigen::Affine3d& ext) { return pose_to_array(ext); });
}

static void populate_intrinsic_blocks(HEParameterBlocks& blocks,
                                      const std::vector<CameraMatrix>& initial_intrinsics) {
    blocks.K.resize(initial_intrinsics.size());
    blocks.dist.resize(initial_intrinsics.size());
    for (size_t i = 0; i < initial_intrinsics.size(); ++i) {
        const auto& intr = initial_intrinsics[i];
        blocks.K[i] = {intr.fx, intr.fy, intr.cx, intr.cy};
        blocks.dist[i] = {0, 0, 0, 0};  // Initialize distortion coefficients to zero
    }
}

static void populate_base_target_block(HEParameterBlocks& blocks,
                                       const Eigen::Affine3d& initial_base_target) {
    blocks.base_target6 = pose_to_array(initial_base_target);
}

static HEParameterBlocks initialize_blocks(
    const std::vector<CameraMatrix>& initial_intrinsics,
    const Eigen::Affine3d& initial_hand_eye,
    const std::vector<Eigen::Affine3d>& initial_extrinsics,
    const Eigen::Affine3d& initial_base_target
) {
    HEParameterBlocks blocks;
    populate_heref6_block(blocks, initial_hand_eye);
    populate_extrinsics_blocks(blocks, initial_extrinsics);
    populate_intrinsic_blocks(blocks, initial_intrinsics);
    populate_base_target_block(blocks, initial_base_target);
    return blocks;
}

static void build_problem(const std::vector<HandEyeObservation>& observations,
                          const HandEyeOptions& opts,
                          HEParameterBlocks& blocks,
                          ceres::Problem& p) {
    for (const auto& obs : observations) {
        const size_t cam = obs.camera_index;
        auto* cost = HandEyeReprojResidual::create(obs.view, obs.base_T_gripper);
        p.AddResidualBlock(cost, new ceres::HuberLoss(1.0),
                           blocks.base_target6.data(),
                           blocks.he_ref6.data(),
                           blocks.ext6[cam].data(),
                           blocks.K[cam].data(),
                           blocks.dist[cam].data());
    }
    // keep identity extrinsic constant
    p.SetParameterBlockConstant(blocks.ext6[0].data());

    const bool single_cam = blocks.K.size() == 1;

    // With a single camera, the hand-eye and target pose cannot be estimated
    // simultaneously.  If both are requested, fix the target pose to the
    // initial guess to remove the gauge freedom.
    if (!opts.optimize_target_pose || (single_cam && opts.optimize_hand_eye)) {
        p.SetParameterBlockConstant(blocks.base_target6.data());
    }
    if (!opts.optimize_hand_eye) {
        p.SetParameterBlockConstant(blocks.he_ref6.data());
    }
    if (!opts.optimize_extrinsics) {
        for (auto& e : blocks.ext6) p.SetParameterBlockConstant(e.data());
    }
    if (!opts.optimize_intrinsics) {
        for (size_t c = 0; c < blocks.K.size(); ++c) {
            p.SetParameterBlockConstant(blocks.K[c].data());
            p.SetParameterBlockConstant(blocks.dist[c].data());
        }
    } else {
        for (size_t c = 0; c < blocks.K.size(); ++c) {
            p.SetParameterLowerBound(blocks.K[c].data(), 0, 0.0);
            p.SetParameterLowerBound(blocks.K[c].data(), 1, 0.0);
        }
        // Anchor scale by fixing fx, fy for reference camera
        std::vector<int> fixed = {0, 1};
        p.SetManifold(blocks.K[0].data(), new ceres::SubsetManifold(4, fixed));
    }
}

static void recover_parameters(const HEParameterBlocks& blocks,
                               HandEyeResult& result) {
    result.hand_eye = array_to_pose(blocks.he_ref6.data());
    result.base_T_target = array_to_pose(blocks.base_target6.data());

    const size_t num_cams = blocks.K.size();
    result.extrinsics.resize(num_cams);
    result.distortions.resize(num_cams);
    result.intrinsics.resize(num_cams);

    for (size_t c = 0; c < num_cams; ++c) {
        result.extrinsics[c] = c > 0 ? array_to_pose(blocks.ext6[c].data()) : Eigen::Affine3d::Identity();
        result.distortions[c] = Eigen::VectorXd::Map(blocks.dist[c].data(), 4);
        result.intrinsics[c] = {blocks.K[c][0], blocks.K[c][1], blocks.K[c][2], blocks.K[c][3]};
    }
}

static double compute_reprojection_error(const std::vector<HandEyeObservation>& observations,
                                         HandEyeResult& result) {
    double ssr = 0.0; size_t total = 0;

    for (const auto& obs : observations) {
        const size_t cam = obs.camera_index;
        const auto& intr = result.intrinsics[cam];
        const Eigen::VectorXd& dist = result.distortions[cam];

        auto camera_T_target = get_camera_T_target(
            result.base_T_target,
            result.hand_eye,
            result.extrinsics[cam],
            obs.base_T_gripper
        );

        std::vector<Observation<double>> o(obs.view.size());
        planar_observables_to_observables(obs.view, o, camera_T_target);

        for (const auto& ob : o) {
            Eigen::Vector2d norm_xy(ob.x, ob.y);
            Eigen::Vector2d distorted = apply_distortion(norm_xy, dist);
            double u = intr.fx * distorted.x() + intr.cx;
            double v = intr.fy * distorted.y() + intr.cy;
            double du = u - ob.u;
            double dv = v - ob.v;
            ssr += du * du + dv * dv;
            total += 2;
        }
    }

    return total ? std::sqrt(ssr / total) : 0.0;
}

static Eigen::MatrixXd compute_covariance(ceres::Problem& p,
                                          HEParameterBlocks& blocks) {
    std::vector<const double*> blocks_list;
    blocks_list.push_back(blocks.base_target6.data());
    blocks_list.push_back(blocks.he_ref6.data());
    for (auto& e : blocks.ext6) blocks_list.push_back(e.data());
    for (auto& k : blocks.K) blocks_list.push_back(k.data());
    for (auto& d : blocks.dist) blocks_list.push_back(d.data());

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
    #if 1
    sopts.linear_solver_type = ceres::DENSE_QR;
    #elif 0
    sopts.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    #else
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

HandEyeResult calibrate_hand_eye(
    const std::vector<HandEyeObservation>& observations,
    const std::vector<CameraMatrix>& initial_intrinsics,
    const Eigen::Affine3d& initial_hand_eye,
    const std::vector<Eigen::Affine3d>& initial_extrinsics,
    const Eigen::Affine3d& initial_base_target,
    const HandEyeOptions& opts
) {
    const size_t num_cams = initial_intrinsics.size();
    if (num_cams == 0) {
        throw std::invalid_argument("No camera intrinsics provided");
    };

    HEParameterBlocks blocks = initialize_blocks(
        initial_intrinsics, initial_hand_eye,
        initial_extrinsics, initial_base_target);

    ceres::Problem p;
    build_problem(observations, opts, blocks, p);

    HandEyeResult result;
    result.summary = solve_problem(p, opts.verbose);

    recover_parameters(blocks, result);
    result.reprojection_error = compute_reprojection_error(observations, result);
    result.covariance = compute_covariance(p, blocks);

    return result;
}

} // namespace vitavision
