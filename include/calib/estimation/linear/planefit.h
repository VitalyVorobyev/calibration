#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

#include "calib/models/cameramodel.h"

namespace calib {

inline auto fit_plane_svd(const std::vector<Eigen::Vector3d>& pts) -> Eigen::Vector4d {
    if (pts.size() < 3) {
        throw std::invalid_argument("Not enough points to fit a plane");
    }
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (const auto& p : pts) centroid += p;
    centroid /= static_cast<double>(pts.size());
    Eigen::MatrixXd A(static_cast<Eigen::Index>(pts.size()), 3);
    for (size_t i = 0; i < pts.size(); ++i) {
        A.row(static_cast<Eigen::Index>(i)) = (pts[i] - centroid).transpose();
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::Vector3d normal = svd.matrixV().col(2);
    double d = -normal.dot(centroid);
    const double nrm = normal.norm();
    return {normal.x() / nrm, normal.y() / nrm, normal.z() / nrm, d / nrm};
}

}  // namespace calib
