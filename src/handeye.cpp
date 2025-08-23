#include "calibration/handeye.h"

// std
#include <numeric>
#include <array>

// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "calibration/planarpose.h"
#include "calibration/distortion.h"

namespace vitavision {

// Utility: average a set of affine transforms (rotation via quaternion averaging)
static Eigen::Affine3d average_affines(const std::vector<Eigen::Affine3d>& poses) {
    if (poses.empty()) return Eigen::Affine3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q_sum(0,0,0,0);
    for (const auto& p : poses) {
        t += p.translation();
        Eigen::Quaterniond q(p.linear());
        if (q_sum.coeffs().dot(q.coeffs()) < 0.0) q.coeffs() *= -1.0;
        q_sum.coeffs() += q.coeffs();
    }
    t /= static_cast<double>(poses.size());
    q_sum.normalize();
    Eigen::Affine3d avg = Eigen::Affine3d::Identity();
    avg.linear() = q_sum.toRotationMatrix();
    avg.translation() = t;
    return avg;
}

// Utility: skew-symmetric matrix from vector
static Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m <<    0, -v.z(),  v.y(),
         v.z(),     0, -v.x(),
        -v.y(),  v.x(),    0;
    return m;
}

static Eigen::Vector3d log_rot(const Eigen::Matrix3d& R) {
    Eigen::AngleAxisd aa(R);
    return aa.axis() * aa.angle();
}

Eigen::Affine3d estimate_hand_eye_initial(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& target_T_camera) {

    const size_t n = base_T_gripper.size();
    if (n < 2 || n != target_T_camera.size()) {
        return Eigen::Affine3d::Identity();
    }

    const size_t m = n - 1; // number of motion pairs
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

    Eigen::MatrixXd C(3*m,3);
    Eigen::VectorXd w(3*m);
    for (size_t i = 0; i < m; ++i) {
        Eigen::Affine3d A = base_T_gripper[i].inverse() * base_T_gripper[i+1];
        Eigen::Affine3d B = target_T_camera[i] * target_T_camera[i+1].inverse();
        C.block<3,3>(3*i,0) = A.rotation() - Eigen::Matrix3d::Identity();
        w.segment<3>(3*i) = R * B.translation() - A.translation();
    }
    Eigen::Vector3d t = C.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(w);

    Eigen::Affine3d X = Eigen::Affine3d::Identity();
    X.linear() = R;
    X.translation() = t;
    return X;
}

struct HandEyeReprojResidual {
    std::vector<PlanarObservation> obs_;
    Eigen::Matrix3d base_R_gripper;
    Eigen::Vector3d base_t_gripper;
    bool use_ext;
    int num_radial;

    HandEyeReprojResidual(const PlanarView& view,
                          const Eigen::Affine3d& base_T_gripper,
                          bool use_ext_, int num_radial_)
        : base_R_gripper(base_T_gripper.rotation()),
          base_t_gripper(base_T_gripper.translation()),
          use_ext(use_ext_), num_radial(num_radial_) {
        obs_.reserve(view.object_xy.size());
        for (size_t i = 0; i < view.object_xy.size(); ++i) {
            obs_.push_back({view.object_xy[i], view.image_uv[i]});
        }
    }

    template <typename T>
    bool operator()(const T* base_target6,
                    const T* he_ref6,
                    const T* ext6,
                    const T* intrinsics,
                    T* residuals) const {
        // base -> target
        Eigen::Matrix<T,3,3> R_bt;
        ceres::AngleAxisToRotationMatrix(base_target6, R_bt.data());
        Eigen::Matrix<T,3,1> t_bt(base_target6[3], base_target6[4], base_target6[5]);

        // gripper -> reference camera
        Eigen::Matrix<T,3,3> R_gr;
        ceres::AngleAxisToRotationMatrix(he_ref6, R_gr.data());
        Eigen::Matrix<T,3,1> t_gr(he_ref6[3], he_ref6[4], he_ref6[5]);

        // reference -> camera extrinsic (optional)
        Eigen::Matrix<T,3,3> R_rc = Eigen::Matrix<T,3,3>::Identity();
        Eigen::Matrix<T,3,1> t_rc(T(0), T(0), T(0));
        if (use_ext) {
            ceres::AngleAxisToRotationMatrix(ext6, R_rc.data());
            t_rc = Eigen::Matrix<T,3,1>(ext6[3], ext6[4], ext6[5]);
        }

        // gripper -> camera
        Eigen::Matrix<T,3,3> R_gc = R_gr * R_rc;
        Eigen::Matrix<T,3,1> t_gc = R_gr * t_rc + t_gr;

        // base -> camera
        Eigen::Matrix<T,3,3> R_bg = base_R_gripper.cast<T>();
        Eigen::Matrix<T,3,1> t_bg = base_t_gripper.cast<T>();

        Eigen::Matrix<T,3,3> R_bc = R_bg * R_gc;
        Eigen::Matrix<T,3,1> t_bc = R_bg * t_gc + t_bg;

        // target -> camera
        Eigen::Matrix<T,3,3> R_tb = R_bt.transpose();
        Eigen::Matrix<T,3,3> R_tc = R_bc * R_tb;
        Eigen::Matrix<T,3,1> t_tc = t_bc - R_tc * t_bt;

        const int N = static_cast<int>(obs_.size());
        static thread_local std::vector<Observation<T>> o;
        if (o.size() != static_cast<size_t>(N)) o.resize(N);
        for (int i = 0; i < N; ++i) {
            const auto& ob = obs_[i];
            Eigen::Matrix<T,3,1> P(T(ob.object_xy.x()), T(ob.object_xy.y()), T(0));
            Eigen::Matrix<T,3,1> Pc = R_tc * P + t_tc;
            o[i] = Observation<T>{Pc.x()/Pc.z(), Pc.y()/Pc.z(), T(ob.image_uv.x()), T(ob.image_uv.y())};
        }

        auto dr = fit_distortion_full(o, intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3], num_radial);
        if (dr) {
            const auto& r = dr->residuals;
            for (int i = 0; i < r.size(); ++i) residuals[i] = r[i];
        } else {
            for (int i = 0; i < N; ++i) {
                const auto& ob = obs_[i];
                Eigen::Matrix<T,3,1> P(T(ob.object_xy.x()), T(ob.object_xy.y()), T(0));
                Eigen::Matrix<T,3,1> Pc = R_tc * P + t_tc;
                T u = intrinsics[0] * (Pc.x() / Pc.z()) + intrinsics[2];
                T v = intrinsics[1] * (Pc.y() / Pc.z()) + intrinsics[3];
                residuals[2*i]   = u - T(ob.image_uv.x());
                residuals[2*i+1] = v - T(ob.image_uv.y());
            }
        }
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
        auto* functor = new HandEyeReprojResidual(obs.view, obs.base_T_gripper, use_ext, 0);
        auto* cost = new ceres::AutoDiffCostFunction<HandEyeReprojResidual,
                                                     ceres::DYNAMIC,
                                                     6,6,6,4>(functor,
                                                              static_cast<int>(obs.view.object_xy.size())*2);
        p.AddResidualBlock(cost, nullptr,
                           blocks.base_target6.data(),
                           blocks.he_ref6.data(),
                           ext_ptr,
                           blocks.K[cam].data());
    }
    // keep identity extrinsic constant
    p.SetParameterBlockConstant(identity_ext6);

    if (!opts.optimize_target_pose) {
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
                                         const HandEyeResult& result) {
    const size_t num_cams = result.intrinsics.size();
    std::vector<Eigen::VectorXd> dists(num_cams, Eigen::VectorXd::Zero(2));
    // Estimate optimal distortion per camera
    for (size_t c = 0; c < num_cams; ++c) {
        std::vector<Observation<double>> o;
        for (const auto& obs : observations) {
            if (obs.camera_index != c) continue;
            Eigen::Matrix3d R_bt = result.base_T_target.rotation();
            Eigen::Vector3d t_bt = result.base_T_target.translation();
            Eigen::Matrix3d R_bg = obs.base_T_gripper.rotation();
            Eigen::Vector3d t_bg = obs.base_T_gripper.translation();
            Eigen::Matrix3d R_gc = result.hand_eye[c].rotation();
            Eigen::Vector3d t_gc = result.hand_eye[c].translation();
            Eigen::Matrix3d R_bc = R_bg * R_gc;
            Eigen::Vector3d t_bc = R_bg * t_gc + t_bg;
            Eigen::Matrix3d R_tb = R_bt.transpose();
            Eigen::Matrix3d R_tc = R_bc * R_tb;
            Eigen::Vector3d t_tc = t_bc - R_tc * t_bt;
            for (size_t i = 0; i < obs.view.object_xy.size(); ++i) {
                Eigen::Vector3d P(obs.view.object_xy[i].x(), obs.view.object_xy[i].y(), 0.0);
                Eigen::Vector3d Pc = R_tc * P + t_tc;
                o.push_back(Observation<double>{Pc.x()/Pc.z(), Pc.y()/Pc.z(),
                                               obs.view.image_uv[i].x(), obs.view.image_uv[i].y()});
            }
        }
        auto dr = fit_distortion(o, result.intrinsics[c].fx, result.intrinsics[c].fy,
                                 result.intrinsics[c].cx, result.intrinsics[c].cy, 0);
        if (dr) dists[c] = dr->distortion;
    }

    double ssr = 0.0; size_t total = 0;
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
            Eigen::Vector2d xyn{Pc.x()/Pc.z(), Pc.y()/Pc.z()};
            Eigen::Vector2d d = apply_distortion(xyn, dists[cam]);
            Eigen::Vector2d pred = result.intrinsics[cam].denormalize(d);
            Eigen::Vector2d diff = pred - uv;
            ssr += diff.squaredNorm();
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

