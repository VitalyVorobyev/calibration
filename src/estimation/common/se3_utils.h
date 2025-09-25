#pragma once

// std
#include <array>
#include <vector>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace calib {

inline Eigen::Matrix3d project_to_so3(const Eigen::Matrix3d& rmtx) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(rmtx, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Matrix3d& umtx = svd.matrixU();
    Eigen::Matrix3d vmtx = svd.matrixV();
    Eigen::Matrix3d sigma = Eigen::Matrix3d::Identity();
    if ((umtx * vmtx.transpose()).determinant() < 0.0) {
        sigma(2, 2) = -1.0;
    }
    return umtx * sigma * vmtx.transpose();
}

inline Eigen::Matrix3d skew(const Eigen::Vector3d& vec) {
    Eigen::Matrix3d skew_mtx;
    skew_mtx << 0, -vec.z(), vec.y(), vec.z(), 0, -vec.x(), -vec.y(), vec.x(), 0;
    return skew_mtx;
}

inline Eigen::Vector3d log_so3(const Eigen::Matrix3d& rot_in) {
    const Eigen::Matrix3d rotation = project_to_so3(rot_in);
    double cos_theta = (rotation.trace() - 1.0) * 0.5;
    cos_theta = std::min(1.0, std::max(-1.0, cos_theta));
    double theta = std::acos(cos_theta);
    if (theta < 1e-12) {
        return Eigen::Vector3d::Zero();
    }
    Eigen::Vector3d wvec;
    wvec << rotation(2, 1) - rotation(1, 2), rotation(0, 2) - rotation(2, 0),
        rotation(1, 0) - rotation(0, 1);
    wvec *= 0.5 / std::sin(theta);
    return wvec * theta;
}

inline Eigen::Matrix3d exp_so3(const Eigen::Vector3d& wvec) {
    double theta = wvec.norm();
    if (theta < 1e-12) {
        return Eigen::Matrix3d::Identity();
    }
    Eigen::Vector3d avec = wvec / theta;
    Eigen::Matrix3d askew = skew(avec);
    return Eigen::Matrix3d::Identity() + std::sin(theta) * askew +
           (1.0 - std::cos(theta)) * (askew * askew);
}

inline Eigen::VectorXd solve_llsq(const Eigen::MatrixXd& amtx, const Eigen::VectorXd& bvec) {
    return amtx.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(bvec);
}

template <class Mat, class Vec>
Eigen::VectorXd ridge_llsq(const Mat& amtx, const Vec& bvec, double lambda = 1e-10) {
    const int ncols = static_cast<int>(amtx.cols());
    return (amtx.transpose() * amtx + lambda * Eigen::MatrixXd::Identity(ncols, ncols))
        .ldlt()
        .solve(amtx.transpose() * bvec);
}

inline std::array<double, 6> pose_to_array(const Eigen::Isometry3d& pose) {
    Eigen::AngleAxisd axisangle(pose.linear());
    return {axisangle.axis().x() * axisangle.angle(),
            axisangle.axis().y() * axisangle.angle(),
            axisangle.axis().z() * axisangle.angle(),
            pose.translation().x(),
            pose.translation().y(),
            pose.translation().z()};
}

inline Eigen::Isometry3d average_affines(const std::vector<Eigen::Isometry3d>& poses) {
    if (poses.empty()) {
        return Eigen::Isometry3d::Identity();
    }
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q_sum(0, 0, 0, 0);
    for (const auto& p : poses) {
        translation += p.translation();
        Eigen::Quaterniond q(p.linear());
        if (q_sum.coeffs().dot(q.coeffs()) < 0.0) {
            q.coeffs() *= -1.0;
        }
        q_sum.coeffs() += q.coeffs();
    }
    translation /= static_cast<double>(poses.size());
    q_sum.normalize();
    Eigen::Isometry3d avg = Eigen::Isometry3d::Identity();
    avg.linear() = q_sum.toRotationMatrix();
    avg.translation() = translation;
    return avg;
}

}  // namespace calib
