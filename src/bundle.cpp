#include "calib/bundle.h"

// std
#include <numeric>
#include <array>

// eigen
#include <Eigen/Geometry>

// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/manifold.h>

#include "calib/planarpose.h"

#include "observationutils.h"
#include "bundleresidual.h"

constexpr double CONVERGENCE_TOLERANCE = 1e-6;
constexpr int MAX_ITERATIONS = 1000;

namespace calib {

struct BundleParamBlocks final {
    std::array<double, 4> b_q_t{};             // target to base
    std::array<double, 3> b_t_t{};
    std::vector<std::array<double, 4>> g_q_c;  // camera to grippes (hand-eye poses)
    std::vector<std::array<double, 3>> g_t_c;
    std::vector<std::array<double, 9>> intr;   // camera matrices and distortions
};

struct ScheimpflugBundleBlocks final {
    std::array<double, 4> b_q_t{};
    std::array<double, 3> b_t_t{};
    std::vector<std::array<double, 4>> g_q_c;
    std::vector<std::array<double, 3>> g_t_c;
    std::vector<std::array<double, 11>> intr;
};

static BundleParamBlocks initialize_blocks(
    const std::vector<Camera>& initial_cameras,
    const std::vector<Eigen::Affine3d>& g_T_c,
    const Eigen::Affine3d& b_T_t
) {
    BundleParamBlocks blocks;
    const size_t ncam = g_T_c.size();

    if (ncam == 0) {
        throw std::runtime_error("No cameras available");
    }

    blocks.g_q_c.resize(ncam);
    blocks.g_t_c.resize(ncam);
    blocks.intr.resize(ncam);

    populate_quat_tran(b_T_t, blocks.b_q_t, blocks.b_t_t);
    for (size_t i = 0; i < ncam; ++i) {
        populate_quat_tran(g_T_c[i], blocks.g_q_c[i], blocks.g_t_c[i]);

        const auto& cam = initial_cameras[i];
        if (cam.distortion.forward.size() != 5) {
            throw std::runtime_error("Invalid distortion parameters");
        }

        blocks.intr[i][0] = cam.K.fx;
        blocks.intr[i][1] = cam.K.fy;
        blocks.intr[i][2] = cam.K.cx;
        blocks.intr[i][3] = cam.K.cy;
        blocks.intr[i][4] = cam.distortion.forward[0];
        blocks.intr[i][5] = cam.distortion.forward[1];
        blocks.intr[i][6] = cam.distortion.forward[2];
        blocks.intr[i][7] = cam.distortion.forward[3];
        blocks.intr[i][8] = cam.distortion.forward[4];
    }

    return blocks;
}

static ScheimpflugBundleBlocks initialize_blocks_scheimpflug(
    const std::vector<ScheimpflugCamera>& initial_cameras,
    const std::vector<Eigen::Affine3d>& g_T_c,
    const Eigen::Affine3d& b_T_t)
{
    ScheimpflugBundleBlocks blocks;
    const size_t ncam = g_T_c.size();

    blocks.g_q_c.resize(ncam);
    blocks.g_t_c.resize(ncam);
    blocks.intr.resize(ncam);

    populate_quat_tran(b_T_t, blocks.b_q_t, blocks.b_t_t);
    for (size_t i = 0; i < ncam; ++i) {
        populate_quat_tran(g_T_c[i], blocks.g_q_c[i], blocks.g_t_c[i]);

        const auto& cam = initial_cameras[i];
        blocks.intr[i][0] = cam.camera.K.fx;
        blocks.intr[i][1] = cam.camera.K.fy;
        blocks.intr[i][2] = cam.camera.K.cx;
        blocks.intr[i][3] = cam.camera.K.cy;
        blocks.intr[i][4] = cam.tau_x;
        blocks.intr[i][5] = cam.tau_y;
        blocks.intr[i][6] = cam.camera.distortion.forward[0];
        blocks.intr[i][7] = cam.camera.distortion.forward[1];
        blocks.intr[i][8] = cam.camera.distortion.forward[2];
        blocks.intr[i][9] = cam.camera.distortion.forward[3];
        blocks.intr[i][10] = cam.camera.distortion.forward[4];
    }
    return blocks;
}

static ceres::Problem build_problem(
    const std::vector<BundleObservation>& observations,
    const BundleOptions& opts,
    BundleParamBlocks& blocks
) {
    ceres::Problem p;
    for (const auto& obs : observations) {
        const size_t cam = obs.camera_index;
        auto* cost = BundleReprojResidual::create(obs.view, obs.b_T_g);
        p.AddResidualBlock(cost, new ceres::HuberLoss(1.0),
                           blocks.b_q_t.data(), blocks.b_t_t.data(),
                           blocks.g_q_c[cam].data(), blocks.g_t_c[cam].data(),
                           blocks.intr[cam].data());
    }

    // set quaternions
    p.SetManifold(blocks.b_q_t.data(), new ceres::QuaternionManifold());
    for (size_t cam = 0; cam < blocks.g_q_c.size(); ++cam) {
        p.SetManifold(blocks.g_q_c[cam].data(), new ceres::QuaternionManifold());
    }

    if (!opts.optimize_target_pose) {
        p.SetParameterBlockConstant(blocks.b_q_t.data());
        p.SetParameterBlockConstant(blocks.b_t_t.data());
    }

    if (!opts.optimize_hand_eye) {
        for (auto& e : blocks.g_q_c) p.SetParameterBlockConstant(e.data());
        for (auto& e : blocks.g_t_c) p.SetParameterBlockConstant(e.data());
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
    }
    return p;
}

static ceres::Problem build_problem_scheimpflug(
    const std::vector<BundleObservation>& observations,
    const BundleOptions& opts,
    ScheimpflugBundleBlocks& blocks
) {
    ceres::Problem p;
    for (const auto& obs : observations) {
        const size_t cam = obs.camera_index;
        p.AddResidualBlock(
            BundleScheimpflugReprojResidual::create(obs.view, obs.b_T_g),
            new ceres::HuberLoss(1.0),
            blocks.b_q_t.data(), blocks.b_t_t.data(),
            blocks.g_q_c[cam].data(), blocks.g_t_c[cam].data(),
            blocks.intr[cam].data());
    }

    p.SetManifold(blocks.b_q_t.data(), new ceres::QuaternionManifold());
    for (size_t cam = 0; cam < blocks.g_q_c.size(); ++cam) {
        p.SetManifold(blocks.g_q_c[cam].data(), new ceres::QuaternionManifold());
    }
    if (!opts.optimize_target_pose) {
        p.SetParameterBlockConstant(blocks.b_q_t.data());
        p.SetParameterBlockConstant(blocks.b_t_t.data());
    }
    if (!opts.optimize_hand_eye){
        for (auto& e : blocks.g_q_c) p.SetParameterBlockConstant(e.data());
        for (auto& e : blocks.g_t_c) p.SetParameterBlockConstant(e.data());
    }
    if (!opts.optimize_intrinsics){
        for (size_t c=0;c<blocks.intr.size();++c)
            p.SetParameterBlockConstant(blocks.intr[c].data());
    } else {
        for (size_t c=0;c<blocks.intr.size();++c){
            p.SetParameterLowerBound(blocks.intr[c].data(),0,0.0);
            p.SetParameterLowerBound(blocks.intr[c].data(),1,0.0);
        }
    }
    return p;
}

static void recover_parameters(
    const BundleParamBlocks& blocks,
    BundleResult& result
) {
    result.b_T_t = restore_pose(blocks.b_q_t, blocks.b_t_t);

    const size_t num_cams = blocks.intr.size();
    result.g_T_c.resize(num_cams);
    result.cameras.resize(num_cams);

    for (size_t c = 0; c < num_cams; ++c) {
        result.g_T_c[c] = restore_pose(blocks.g_q_c[c], blocks.g_t_c[c]);
        const auto& i = blocks.intr[c];
        CameraMatrix K{i[0], i[1], i[2], i[3]};
        Eigen::VectorXd dist(5);
        dist << i[4], i[5], i[6], i[7], i[8];
        result.cameras[c] = Camera(K, dist);
    }
}

static void recover_parameters(const ScheimpflugBundleBlocks& blocks,
                               ScheimpflugBundleResult& result){
    result.b_T_t = restore_pose(blocks.b_q_t, blocks.b_t_t);
    const size_t num_cams = blocks.intr.size();
    result.g_T_c.resize(num_cams);
    result.cameras.resize(num_cams);

    for (size_t c = 0; c < num_cams; ++c) {
        result.g_T_c[c] = restore_pose(blocks.g_q_c[c], blocks.g_t_c[c]);
        const auto& i = blocks.intr[c];
        CameraMatrix K{ i[0], i[1], i[2], i[3] };
        Eigen::VectorXd dist(5);
        dist << i[6], i[7], i[8], i[9], i[10];
        Camera cam(K, dist);
        result.cameras[c] = ScheimpflugCamera(cam, i[4], i[5]);
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

        auto c_T_t = get_camera_T_target(result.b_T_t, result.g_T_c[cam_idx], obs.b_T_g);

        Eigen::Vector3d P;
        for (const auto& ob : obs.view) {
            P << ob.object_xy.x(), ob.object_xy.y(), 0;
            P = c_T_t * P;
            auto uv_hat = cam.project(P);
            auto duv = uv_hat - ob.image_uv;
            ssr += duv.squaredNorm();
            total += 2;
        }
    }

    return total ? std::sqrt(ssr / total) : 0.0;
}

static double compute_reprojection_error_scheimpflug(
    const std::vector<BundleObservation>& observations,
    ScheimpflugBundleResult& result)
{
    double ssr = 0.0;
    size_t total = 0;
    for (const auto& obs : observations){
        const size_t cam_idx = obs.camera_index;
        const auto& sc = result.cameras[cam_idx];
        auto c_T_t = get_camera_T_target(result.b_T_t, result.g_T_c[cam_idx], obs.b_T_g);

        Eigen::Vector3d P;
        for (const auto& ob : obs.view){
            P << ob.object_xy.x(), ob.object_xy.y(), 0;
            P = c_T_t * P;
            auto uv_hat = sc.project(P);
            auto delta_uv = uv_hat - ob.image_uv;
            ssr += delta_uv.squaredNorm();
            total+=2;
        }
    }
    return total ? std::sqrt(ssr / total) : 0.0;
}

static Eigen::MatrixXd compute_covariance(ceres::Problem& p,
                                          BundleParamBlocks& blocks) {
    std::vector<const double*> blocks_list;
    blocks_list.push_back(blocks.b_q_t.data());
    blocks_list.push_back(blocks.b_t_t.data());
    for (auto& e : blocks.g_q_c) blocks_list.push_back(e.data());
    for (auto& e : blocks.g_t_c) blocks_list.push_back(e.data());
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

static Eigen::MatrixXd compute_covariance(ceres::Problem& p,
                                          ScheimpflugBundleBlocks& blocks) {
    std::vector<const double*> blocks_list;
    blocks_list.push_back(blocks.b_q_t.data());
    blocks_list.push_back(blocks.b_t_t.data());
    for (auto& e : blocks.g_q_c) blocks_list.push_back(e.data());
    for (auto& e : blocks.g_t_c) blocks_list.push_back(e.data());
    for (auto& e : blocks.intr) blocks_list.push_back(e.data());

    ceres::Covariance::Options cov_opts; ceres::Covariance cov(cov_opts);
    if (!cov.Compute(blocks_list, &p)) return Eigen::MatrixXd();
    std::vector<int> sizes(blocks_list.size());
    for (size_t i = 0; i < blocks_list.size(); ++i) {
        sizes[i] = p.ParameterBlockSize(blocks_list[i]);
    }
    int dim=0; for (int s:sizes) dim+=s;
    Eigen::MatrixXd cov_mat = Eigen::MatrixXd::Zero(dim,dim);
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
    const std::vector<Eigen::Affine3d>& init_g_T_c,
    const Eigen::Affine3d& init_b_T_t,
    const BundleOptions& opts
) {
    const size_t num_cams = initial_cameras.size();
    if (num_cams == 0) {
        throw std::invalid_argument("No camera intrinsics provided");
    };

    BundleParamBlocks blocks = initialize_blocks(
        initial_cameras, init_g_T_c, init_b_T_t);

    ceres::Problem p = build_problem(observations, opts, blocks);

    BundleResult result;
    result.report = solve_problem(p, opts.verbose);

    recover_parameters(blocks, result);
    result.reprojection_error = compute_reprojection_error(observations, result);
    result.covariance = compute_covariance(p, blocks);

    return result;
}

ScheimpflugBundleResult optimize_bundle_scheimpflug(
    const std::vector<BundleObservation>& observations,
    const std::vector<ScheimpflugCamera>& initial_cameras,
    const std::vector<Eigen::Affine3d>& init_g_T_c,
    const Eigen::Affine3d& init_b_T_t,
    const BundleOptions& opts)
{
    const size_t num_cams = initial_cameras.size();
    if (num_cams == 0) {
        throw std::invalid_argument("No camera intrinsics provided");
    }
    if (init_g_T_c.size() != num_cams) {
        throw std::invalid_argument("Invalid initial transforms provided");
    }

    ScheimpflugBundleBlocks blocks = initialize_blocks_scheimpflug(
        initial_cameras, init_g_T_c, init_b_T_t);

    ceres::Problem p = build_problem_scheimpflug(observations, opts, blocks);

    ScheimpflugBundleResult result; result.report = solve_problem(p, opts.verbose);
    recover_parameters(blocks, result);
    result.reprojection_error = compute_reprojection_error_scheimpflug(observations, result);
    result.covariance = compute_covariance(p, blocks);
    return result;
}

}  // namespace calib
