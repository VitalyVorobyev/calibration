#include "planeestimator.h"

// std
#include <numeric>

// eigen
#include <Eigen/Dense>

namespace calib {

using Datum = PlaneEstimator::Datum;
using Model = PlaneEstimator::Model;

auto PlaneEstimator::fit(const std::vector<Datum>& data, std::span<const int> sample)
    -> std::optional<Model> {
    if (sample.size() < k_min_samples) {
        return std::nullopt;
    }

    const Datum& p0 = data[sample[0]];
    const Datum& p1 = data[sample[1]];
    const Datum& p2 = data[sample[2]];

    Eigen::Vector3d n = (p1 - p0).cross(p2 - p0);
    double norm = n.norm();
    if (norm < 1e-12) {
        return std::nullopt;  // near-degenerate
    }
    n /= norm;
    double d = -n.dot(p0);
    return Model{n.x(), n.y(), n.z(), d};
}

auto PlaneEstimator::residual(const Model& plane, const Datum& p) -> double {
    return std::abs(plane.head<3>().dot(p) + plane[3]);
}

auto PlaneEstimator::refit(const std::vector<Datum>& data, std::span<const int> inliers)
    -> std::optional<Model> {
    if (inliers.size() < k_min_samples) {
        return std::nullopt;
    }

    std::vector<Datum> pts;
    pts.reserve(inliers.size());
    for (int idx : inliers) {
        pts.push_back(data[idx]);
    }

    Eigen::Vector3d centroid = std::accumulate(pts.begin(), pts.end(), Eigen::Vector3d::Zero());
    centroid /= static_cast<double>(pts.size());

    Eigen::MatrixXd A(pts.size(), 3);
    for (size_t i = 0; i < pts.size(); ++i) {
        A.row(static_cast<Eigen::Index>(i)) = (pts[i] - centroid).transpose();
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::Vector3d n = svd.matrixV().col(2);
    n.normalize();
    double d = -n.dot(centroid);
    return Model{n.x(), n.y(), n.z(), d};
}

auto PlaneEstimator::is_degenerate(const std::vector<Datum>& data, std::span<const int> sample)
    -> bool {
    if (sample.size() < k_min_samples) {
        return true;
    }
    const Datum& p0 = data[sample[0]];
    const Datum& p1 = data[sample[1]];
    const Datum& p2 = data[sample[2]];
    Eigen::Vector3d n = (p1 - p0).cross(p2 - p0);
    return n.squaredNorm() < 1e-12;
}

}  // namespace calib
