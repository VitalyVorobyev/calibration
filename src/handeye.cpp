#include "calibration/handeye.h"

// std
#include <numeric>
#include <array>

// eigen
#include <Eigen/Geometry>

// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "calibration/planarpose.h"

#include "observationutils.h"

namespace vitavision {

static Eigen::Vector3d log_rot(const Eigen::Matrix3d& R) {
    Eigen::AngleAxisd aa(R);
    return aa.axis() * aa.angle();
}

static Eigen::Matrix3d estimate_hand_eye_rotation(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& target_T_camera
) {
    const size_t n = base_T_gripper.size();
    const size_t m = n - 1;  // number of motion pairs
    Eigen::MatrixXd M(3*m, 3);
    Eigen::VectorXd d(3*m);

    for (size_t i = 0; i < m; ++i) {
        Eigen::Affine3d A = base_T_gripper[i].inverse() * base_T_gripper[i+1];
        Eigen::Affine3d B = target_T_camera[i] * target_T_camera[i+1].inverse();

        Eigen::Vector3d alpha = log_rot(A.rotation());
        Eigen::Vector3d beta  = log_rot(B.rotation());
        M.block<3,3>(3*i,0) = skew(alpha + beta);
        d.segment<3>(3*i) = beta - alpha;
    }

    Eigen::Vector3d r = M.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(d);
    double angle = r.norm();
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    if (angle > 1e-12) {
        R = Eigen::AngleAxisd(angle, r.normalized()).toRotationMatrix();
    }

    return R;
}

static Eigen::Vector3d estimate_hand_eye_translation(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& target_T_camera,
    const Eigen::Matrix3d& R
) {
    const size_t n = base_T_gripper.size();
    const size_t m = n - 1;  // number of motion pairs

    Eigen::MatrixXd C(3*m,3);
    Eigen::VectorXd w(3*m);
    for (size_t i = 0; i < m; ++i) {
        Eigen::Affine3d A = base_T_gripper[i].inverse() * base_T_gripper[i+1];
        Eigen::Affine3d B = target_T_camera[i] * target_T_camera[i+1].inverse();
        C.block<3,3>(3*i,0) = A.rotation() - Eigen::Matrix3d::Identity();
        w.segment<3>(3*i) = R * B.translation() - A.translation();
    }
    Eigen::Vector3d t = C.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(w);
    return t;
}

Eigen::Affine3d estimate_hand_eye_initial(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& target_T_camera) {

    const size_t n = base_T_gripper.size();
    if (n < 2 || n != target_T_camera.size()) {
        std::cerr << "Insufficient data for initial hand-eye estimate\n";
        return Eigen::Affine3d::Identity();
    }
    auto R = estimate_hand_eye_rotation(base_T_gripper, target_T_camera);
    auto t = estimate_hand_eye_translation(base_T_gripper, target_T_camera, R);

    Eigen::Affine3d X = Eigen::Affine3d::Identity();
    X.linear() = R;
    X.translation() = t;
    return X;
}

struct HandEyeReprojResidual final {
    PlanarView view;
    Eigen::Affine3d base_to_gripper;
    bool use_ext;
    int num_radial;

    HandEyeReprojResidual(PlanarView v,
                          const Eigen::Affine3d& base_T_gripper,
                          bool use_ext_,
                          int num_radial_=2)
        : view(std::move(v)),
          base_to_gripper(base_T_gripper),
          use_ext(use_ext_),
          num_radial(num_radial_) {}

    template <typename T>
    bool operator()(const T* base_target6,
                    const T* he_ref6,
                    const T* ext6,
                    const T* intrinsics,
                    T* residuals) const {
        // base -> target
        auto pose_bt = pose2affine(base_target6);
        #if 0
        Eigen::Matrix<T,3,3> R_bt;
        ceres::AngleAxisToRotationMatrix(base_target6, R_bt.data());
        Eigen::Matrix<T,3,1> t_bt(base_target6[3], base_target6[4], base_target6[5]);
        #endif

        // gripper -> reference camera
        auto pose_gr = pose2affine(he_ref6);
        #if 0
        Eigen::Matrix<T,3,3> R_gr;
        ceres::AngleAxisToRotationMatrix(he_ref6, R_gr.data());
        Eigen::Matrix<T,3,1> t_gr(he_ref6[3], he_ref6[4], he_ref6[5]);
        #endif

        // reference -> camera extrinsic (optional)
        auto pose_rc = use_ext ? pose2affine(ext6) : Eigen::Transform<T, 3, Eigen::Affine>::Identity();
        #if 0
        Eigen::Matrix<T,3,3> R_rc = Eigen::Matrix<T,3,3>::Identity();
        Eigen::Matrix<T,3,1> t_rc(T(0), T(0), T(0));
        if (use_ext) {
            ceres::AngleAxisToRotationMatrix(ext6, R_rc.data());
            t_rc = Eigen::Matrix<T,3,1>(ext6[3], ext6[4], ext6[5]);
        }
        #endif

        // gripper -> camera
        auto pose_gc = pose_gr * pose_rc;
        #if 0
        Eigen::Matrix<T,3,3> R_gc = R_gr * R_rc;
        Eigen::Matrix<T,3,1> t_gc = R_gr * t_rc + t_gr;
        #endif

        // base -> camera
        auto pose_bc = base_to_gripper.template cast<T>() * pose_gc;

        #if 0
        Eigen::Matrix<T,3,3> R_bg = base_R_gripper.cast<T>();
        Eigen::Matrix<T,3,1> t_bg = base_t_gripper.cast<T>();

        Eigen::Matrix<T,3,3> R_bc = R_bg * R_gc;
        Eigen::Matrix<T,3,1> t_bc = R_bg * t_gc + t_bg;
        #endif

        // target -> camera
        auto pose_tc = pose_bc * pose_bt.inverse();

        #if 0
        Eigen::Matrix<T,3,3> R_tb = R_bt.transpose();
        Eigen::Matrix<T,3,3> R_tc = R_bc * R_tb;
        Eigen::Matrix<T,3,1> t_tc = t_bc - R_tc * t_bt;
        #endif

        const int N = static_cast<int>(view.object_xy.size());
        static thread_local std::vector<Observation<T>> o;
        #if 1
        planar_observables_to_observables(view.observations, o, pose_tc);
        #else
        if (o.size() != static_cast<size_t>(N)) o.resize(N);
        for (int i = 0; i < N; ++i) {
            Eigen::Matrix<T,3,1> P{T(view.object_xy[i].x()), T(view.object_xy[i].y()), T(0)};
            Eigen::Matrix<T,3,1> Pc = R_tc * P + t_tc;
            T xn = Pc.x() / Pc.z();
            T yn = Pc.y() / Pc.z();
            o[i] = Observation<T>{xn, yn,
                                  T(view.image_uv[i].x()),
                                  T(view.image_uv[i].y())};
        }
        #endif

        auto dr = fit_distortion_full(o, intrinsics[0], intrinsics[1],
                                      intrinsics[2], intrinsics[3],
                                      num_radial);
        if (!dr) return false;
        const auto& r = dr->residuals;
        for (int i = 0; i < r.size(); ++i) residuals[i] = r[i];
        return true;
    }
};

struct HEParameterBlocks final {
    std::array<double,6> base_target6{};
    std::array<double,6> he_ref6{};
    std::vector<std::array<double,6>> ext6;
    std::vector<std::array<double,4>> K;
};

static HEParameterBlocks initialise_blocks(
    const std::vector<HandEyeObservation>& observations,
    const std::vector<CameraMatrix>& initial_intrinsics,
    const Eigen::Affine3d& initial_hand_eye,
    const std::vector<Eigen::Affine3d>& initial_extrinsics,
    const Eigen::Affine3d& initial_base_target,
    HandEyeResult& result) {

    const size_t num_cams = initial_intrinsics.size();
    HEParameterBlocks blocks;
    blocks.ext6.resize(num_cams > 0 ? num_cams - 1 : 0);
    blocks.K.resize(num_cams);

    result.hand_eye.resize(num_cams);
    result.extrinsics.resize(num_cams > 0 ? num_cams - 1 : 0);
    result.distortions.resize(num_cams);

    // he_ref
    Eigen::AngleAxisd aa_hr(initial_hand_eye.rotation());
    blocks.he_ref6 = {aa_hr.axis().x()*aa_hr.angle(), aa_hr.axis().y()*aa_hr.angle(),
                     aa_hr.axis().z()*aa_hr.angle(),
                     initial_hand_eye.translation().x(),
                     initial_hand_eye.translation().y(),
                     initial_hand_eye.translation().z()};
    result.hand_eye[0] = initial_hand_eye;

    // extrinsics and other hand_eye transforms
    for (size_t c = 1; c < num_cams; ++c) {
        Eigen::Affine3d ext = (c-1 < initial_extrinsics.size()) ?
            initial_extrinsics[c-1] : Eigen::Affine3d::Identity();
        result.extrinsics[c-1] = ext;
        result.hand_eye[c] = initial_hand_eye * ext;
        Eigen::AngleAxisd aa(ext.rotation());
        blocks.ext6[c-1] = {aa.axis().x()*aa.angle(), aa.axis().y()*aa.angle(),
                            aa.axis().z()*aa.angle(),
                            ext.translation().x(), ext.translation().y(), ext.translation().z()};
    }

    // Intrinsics
    for (size_t c = 0; c < num_cams; ++c) {
        blocks.K[c] = {initial_intrinsics[c].fx, initial_intrinsics[c].fy,
                       initial_intrinsics[c].cx, initial_intrinsics[c].cy};
    }

    // Initial base->target estimate
    Eigen::Affine3d bt_init = initial_base_target;
    std::vector<Eigen::Affine3d> bt_estimates;
    bt_estimates.reserve(observations.size());
    for (const auto& obs : observations) {
        const size_t cam = obs.camera_index;
        if (obs.view.object_xy.size() < 4) continue;
        Eigen::Affine3d cam_T_target = estimate_planar_pose_dlt(
            obs.view.object_xy, obs.view.image_uv, initial_intrinsics[cam]);
        bt_estimates.push_back(obs.base_T_gripper * result.hand_eye[cam] * cam_T_target.inverse());
    }
    if (!bt_estimates.empty()) bt_init = average_affines(bt_estimates);

    Eigen::AngleAxisd aa_bt(bt_init.rotation());
    blocks.base_target6 = {aa_bt.axis().x()*aa_bt.angle(), aa_bt.axis().y()*aa_bt.angle(),
                           aa_bt.axis().z()*aa_bt.angle(),
                           bt_init.translation().x(), bt_init.translation().y(), bt_init.translation().z()};

    return blocks;
}

static void build_problem(const std::vector<HandEyeObservation>& observations,
                          const HandEyeOptions& opts,
                          HEParameterBlocks& blocks,
                          ceres::Problem& p) {
    double identity_ext6[6] = {0,0,0,0,0,0};
    for (const auto& obs : observations) {
        const size_t cam = obs.camera_index;
        bool use_ext = cam > 0;
        double* ext_ptr = use_ext ? blocks.ext6[cam-1].data() : identity_ext6;
        auto* cost = new ceres::AutoDiffCostFunction<HandEyeReprojResidual, ceres::DYNAMIC,6,6,6,4>(
            new HandEyeReprojResidual(obs.view, obs.base_T_gripper, use_ext),
            static_cast<int>(obs.view.object_xy.size())*2);
        p.AddResidualBlock(cost, nullptr,
                           blocks.base_target6.data(),
                           blocks.he_ref6.data(),
                           ext_ptr,
                           blocks.K[cam].data());
    }
    // keep identity extrinsic constant
    p.SetParameterBlockConstant(identity_ext6);

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
        for (auto& k : blocks.K) p.SetParameterBlockConstant(k.data());
    }
}

static void recover_parameters(const HEParameterBlocks& blocks,
                               HandEyeResult& result) {
    const size_t num_cams = blocks.K.size();

    Eigen::Matrix3d R_bt;
    ceres::AngleAxisToRotationMatrix(blocks.base_target6.data(), R_bt.data());
    result.base_T_target.linear() = R_bt;
    result.base_T_target.translation() = Eigen::Vector3d(blocks.base_target6[3], blocks.base_target6[4], blocks.base_target6[5]);

    Eigen::Matrix3d R_hr;
    ceres::AngleAxisToRotationMatrix(blocks.he_ref6.data(), R_hr.data());
    Eigen::Vector3d t_hr(blocks.he_ref6[3], blocks.he_ref6[4], blocks.he_ref6[5]);
    Eigen::Affine3d he_ref = Eigen::Affine3d::Identity();
    he_ref.linear() = R_hr;
    he_ref.translation() = t_hr;
    result.hand_eye[0] = he_ref;

    for (size_t c = 1; c < num_cams; ++c) {
        Eigen::Matrix3d R_ext;
        ceres::AngleAxisToRotationMatrix(blocks.ext6[c-1].data(), R_ext.data());
        Eigen::Vector3d t_ext(blocks.ext6[c-1][3], blocks.ext6[c-1][4], blocks.ext6[c-1][5]);
        Eigen::Affine3d ext = Eigen::Affine3d::Identity();
        ext.linear() = R_ext;
        ext.translation() = t_ext;
        result.extrinsics[c-1] = ext;
        result.hand_eye[c] = he_ref * ext;
    }

    for (size_t c = 0; c < num_cams; ++c) {
        result.intrinsics[c] = {blocks.K[c][0], blocks.K[c][1], blocks.K[c][2], blocks.K[c][3]};
    }
}

static double compute_reprojection_error(const std::vector<HandEyeObservation>& observations,
                                         HandEyeResult& result) {
    const size_t num_cams = result.intrinsics.size();
    std::vector<std::vector<Observation<double>>> per_cam_obs(num_cams);

    for (const auto& obs : observations) {
        const size_t cam = obs.camera_index;
        Eigen::Matrix3d R_bt = result.base_T_target.rotation();
        Eigen::Vector3d t_bt = result.base_T_target.translation();
        Eigen::Matrix3d R_bg = obs.base_T_gripper.rotation();
        Eigen::Vector3d t_bg = obs.base_T_gripper.translation();
        Eigen::Matrix3d R_gc = result.hand_eye[cam].rotation();
        Eigen::Vector3d t_gc = result.hand_eye[cam].translation();
        Eigen::Matrix3d R_bc = R_bg * R_gc;
        Eigen::Vector3d t_bc = R_bg * t_gc + t_bg;
        Eigen::Matrix3d R_tb = R_bt.transpose();
        Eigen::Matrix3d R_tc = R_bc * R_tb;
        Eigen::Vector3d t_tc = t_bc - R_tc * t_bt;
        for (size_t i = 0; i < obs.view.object_xy.size(); ++i) {
            const auto& xy = obs.view.object_xy[i];
            const auto& uv = obs.view.image_uv[i];
            Eigen::Vector3d P(xy.x(), xy.y(), 0.0);
            Eigen::Vector3d Pc = R_tc * P + t_tc;
            double xn = Pc.x() / Pc.z();
            double yn = Pc.y() / Pc.z();
            per_cam_obs[cam].push_back(Observation<double>{xn, yn, uv.x(), uv.y()});
        }
    }

    result.distortions.assign(num_cams, Eigen::VectorXd());
    double ssr = 0.0; size_t total = 0;
    const int num_radial = 2;
    for (size_t c = 0; c < num_cams; ++c) {
        if (per_cam_obs[c].empty()) continue;
        auto dr = fit_distortion_full(per_cam_obs[c],
                                      result.intrinsics[c].fx,
                                      result.intrinsics[c].fy,
                                      result.intrinsics[c].cx,
                                      result.intrinsics[c].cy,
                                      num_radial);
        if (dr) {
            result.distortions[c] = dr->distortion;
            ssr += dr->residuals.squaredNorm();
            total += dr->residuals.size();
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
    ceres::Covariance::Options cov_opts;
    ceres::Covariance cov(cov_opts);
    if (!cov.Compute(blocks_list, &p)) return Eigen::MatrixXd();
    const size_t dim = blocks_list.size() * 6;
    Eigen::MatrixXd cov_mat = Eigen::MatrixXd::Zero(dim, dim);
    cov.GetCovarianceMatrix(blocks_list, cov_mat.data());
    return cov_mat;
}

HandEyeResult calibrate_hand_eye(
    const std::vector<HandEyeObservation>& observations,
    const std::vector<CameraMatrix>& initial_intrinsics,
    const Eigen::Affine3d& initial_hand_eye,
    const std::vector<Eigen::Affine3d>& initial_extrinsics,
    const Eigen::Affine3d& initial_base_target,
    const HandEyeOptions& opts) {

    HandEyeResult result;
    const size_t num_cams = initial_intrinsics.size();
    if (num_cams == 0) return result;

    result.intrinsics = initial_intrinsics;

    HEParameterBlocks blocks = initialise_blocks(
        observations, initial_intrinsics, initial_hand_eye,
        initial_extrinsics, initial_base_target, result);

    ceres::Problem p;
    build_problem(observations, opts, blocks, p);

    ceres::Solver::Options sopts;
    sopts.linear_solver_type = ceres::DENSE_QR;
    sopts.minimizer_progress_to_stdout = opts.verbose;

    constexpr double eps = 1e-6;
    sopts.function_tolerance = eps;
    sopts.gradient_tolerance = eps;
    sopts.parameter_tolerance = eps;
    sopts.max_num_iterations = 1000;

    ceres::Solver::Summary summary;
    ceres::Solve(sopts, &p, &summary);
    result.summary = summary.BriefReport();

    recover_parameters(blocks, result);
    result.reprojection_error = compute_reprojection_error(observations, result);
    result.covariance = compute_covariance(p, blocks);

    return result;
}

} // namespace vitavision
