#include "calibration/handeye.h"

// std
#include <numeric>

// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "calibration/planarpose.h"

namespace vitavision {

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
    Eigen::Vector2d obj_xy;
    Eigen::Vector2d img_uv;
    Eigen::Matrix3d base_R_gripper;
    Eigen::Vector3d base_t_gripper;

    HandEyeReprojResidual(const Eigen::Vector2d& xy,
                          const Eigen::Vector2d& uv,
                          const Eigen::Affine3d& base_T_gripper)
        : obj_xy(xy), img_uv(uv),
          base_R_gripper(base_T_gripper.rotation()),
          base_t_gripper(base_T_gripper.translation()) {}

    template <typename T>
    bool operator()(const T* base_target6,
                    const T* hand_eye6,
                    const T* intrinsics,
                    T* residuals) const {
        // base -> target
        Eigen::Matrix<T,3,3> R_bt;
        ceres::AngleAxisToRotationMatrix(base_target6, R_bt.data());
        Eigen::Matrix<T,3,1> t_bt(base_target6[3], base_target6[4], base_target6[5]);

        // hand eye (gripper -> camera)
        Eigen::Matrix<T,3,3> R_gc;
        ceres::AngleAxisToRotationMatrix(hand_eye6, R_gc.data());
        Eigen::Matrix<T,3,1> t_gc(hand_eye6[3], hand_eye6[4], hand_eye6[5]);

        // base -> camera
        Eigen::Matrix<T,3,3> R_bg = base_R_gripper.cast<T>();
        Eigen::Matrix<T,3,1> t_bg = base_t_gripper.cast<T>();

        Eigen::Matrix<T,3,3> R_bc = R_bg * R_gc;
        Eigen::Matrix<T,3,1> t_bc = R_bg * t_gc + t_bg;

        // target -> camera
        Eigen::Matrix<T,3,3> R_tc = R_bt.transpose() * R_bc;
        Eigen::Matrix<T,3,1> t_tc = R_bt.transpose() * (t_bc - t_bt);

        // point on plane in target frame
        Eigen::Matrix<T,3,1> P(T(obj_xy.x()), T(obj_xy.y()), T(0));
        Eigen::Matrix<T,3,1> Pc = R_tc * P + t_tc;

        T u = intrinsics[0] * (Pc.x() / Pc.z()) + intrinsics[2];
        T v = intrinsics[1] * (Pc.y() / Pc.z()) + intrinsics[3];

        residuals[0] = u - T(img_uv.x());
        residuals[1] = v - T(img_uv.y());
        return true;
    }
};

HandEyeResult calibrate_hand_eye(
    const std::vector<HandEyeObservation>& observations,
    const std::vector<CameraMatrix>& initial_intrinsics,
    const std::vector<Eigen::Affine3d>& initial_hand_eye,
    const HandEyeOptions& opts
) {
    HandEyeResult result;
    const size_t num_cams = initial_intrinsics.size();
    if (num_cams == 0) return result;

    result.intrinsics = initial_intrinsics;
    result.hand_eye = initial_hand_eye;

    // Parameter blocks
    double base_target6[6] = {0,0,0,0,0,0};
    std::vector<std::array<double,6>> he6(num_cams);
    std::vector<std::array<double,4>> K(num_cams);

    for (size_t c = 0; c < num_cams; ++c) {
        Eigen::AngleAxisd aa(result.hand_eye[c].rotation());
        he6[c][0] = aa.axis().x()*aa.angle();
        he6[c][1] = aa.axis().y()*aa.angle();
        he6[c][2] = aa.axis().z()*aa.angle();
        he6[c][3] = result.hand_eye[c].translation().x();
        he6[c][4] = result.hand_eye[c].translation().y();
        he6[c][5] = result.hand_eye[c].translation().z();

        K[c][0] = result.intrinsics[c].fx;
        K[c][1] = result.intrinsics[c].fy;
        K[c][2] = result.intrinsics[c].cx;
        K[c][3] = result.intrinsics[c].cy;
    }

    ceres::Problem p;
    for (const auto& obs : observations) {
        const size_t cam = obs.camera_index;
        for (size_t i = 0; i < obs.view.object_xy.size(); ++i) {
            const auto& xy = obs.view.object_xy[i];
            const auto& uv = obs.view.image_uv[i];
            auto* cost = new ceres::AutoDiffCostFunction<HandEyeReprojResidual,2,6,6,4>(
                new HandEyeReprojResidual(xy, uv, obs.base_T_gripper));
            p.AddResidualBlock(cost, nullptr, base_target6, he6[cam].data(), K[cam].data());
        }
    }

    if (!opts.optimize_target_pose) {
        p.SetParameterBlockConstant(base_target6);
    }
    if (!opts.optimize_hand_eye) {
        for (size_t c = 0; c < num_cams; ++c) {
            p.SetParameterBlockConstant(he6[c].data());
        }
    }
    if (!opts.optimize_intrinsics) {
        for (size_t c = 0; c < num_cams; ++c) {
            p.SetParameterBlockConstant(K[c].data());
        }
    }

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

    // Recover parameters
    Eigen::Matrix3d R_bt;
    ceres::AngleAxisToRotationMatrix(base_target6, R_bt.data());
    result.base_T_target.linear() = R_bt;
    result.base_T_target.translation() = Eigen::Vector3d(base_target6[3], base_target6[4], base_target6[5]);

    double ssr = 0.0; size_t total = 0;
    for (size_t cam = 0; cam < num_cams; ++cam) {
        Eigen::Matrix3d R_gc;
        ceres::AngleAxisToRotationMatrix(he6[cam].data(), R_gc.data());
        result.hand_eye[cam].linear() = R_gc;
        result.hand_eye[cam].translation() = Eigen::Vector3d(he6[cam][3], he6[cam][4], he6[cam][5]);

        result.intrinsics[cam] = {K[cam][0], K[cam][1], K[cam][2], K[cam][3]};
    }

    // compute reprojection error
    for (const auto& obs : observations) {
        const size_t cam = obs.camera_index;
        for (size_t i = 0; i < obs.view.object_xy.size(); ++i) {
            const auto& xy = obs.view.object_xy[i];
            const auto& uv = obs.view.image_uv[i];

            Eigen::Matrix3d R_bt = result.base_T_target.rotation();
            Eigen::Vector3d t_bt = result.base_T_target.translation();
            Eigen::Matrix3d R_bg = obs.base_T_gripper.rotation();
            Eigen::Vector3d t_bg = obs.base_T_gripper.translation();
            Eigen::Matrix3d R_gc = result.hand_eye[cam].rotation();
            Eigen::Vector3d t_gc = result.hand_eye[cam].translation();
            Eigen::Matrix3d R_bc = R_bg * R_gc;
            Eigen::Vector3d t_bc = R_bg * t_gc + t_bg;
            Eigen::Matrix3d R_tc = R_bt.transpose() * R_bc;
            Eigen::Vector3d t_tc = R_bt.transpose() * (t_bc - t_bt);

            Eigen::Vector3d P(xy.x(), xy.y(), 0.0);
            Eigen::Vector3d Pc = R_tc * P + t_tc;
            double u = result.intrinsics[cam].fx * (Pc.x() / Pc.z()) + result.intrinsics[cam].cx;
            double v = result.intrinsics[cam].fy * (Pc.y() / Pc.z()) + result.intrinsics[cam].cy;
            double du = u - uv.x();
            double dv = v - uv.y();
            ssr += du*du + dv*dv;
            total += 1;
        }
    }
    if (total) result.reprojection_error = std::sqrt(ssr / (2*total));

    return result;
}

} // namespace vitavision

