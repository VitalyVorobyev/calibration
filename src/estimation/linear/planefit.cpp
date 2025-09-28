#include "calib/estimation/linear/planefit.h"

#include <Eigen/Geometry>
#include <optional>
#include <span>

namespace {

struct PlaneRansacEstimator final {
    using Datum = Eigen::Vector3d;
    using Model = Eigen::Vector4d;
    static constexpr std::size_t k_min_samples = 3;

    static auto fit(const std::vector<Datum>& data, std::span<const int> sample)
        -> std::optional<Model> {
        if (sample.size() < k_min_samples) {
            return std::nullopt;
        }
        const Eigen::Vector3d& p0 = data[static_cast<std::size_t>(sample[0])];
        const Eigen::Vector3d& p1 = data[static_cast<std::size_t>(sample[1])];
        const Eigen::Vector3d& p2 = data[static_cast<std::size_t>(sample[2])];

        const Eigen::Vector3d v1 = p1 - p0;
        const Eigen::Vector3d v2 = p2 - p0;
        Eigen::Vector3d normal = v1.cross(v2);
        const double norm = normal.norm();
        if (norm < 1e-12) {
            return std::nullopt;
        }
        normal /= norm;
        const double d = -normal.dot(p0);
        return Model{normal.x(), normal.y(), normal.z(), d};
    }

    static auto residual(const Model& plane, const Datum& point) -> double {
        const Eigen::Vector3d normal = plane.head<3>();
        return std::abs(normal.dot(point) + plane[3]);
    }

    static auto refit(const std::vector<Datum>& data, std::span<const int> inliers)
        -> std::optional<Model> {
        if (inliers.size() < static_cast<std::ptrdiff_t>(k_min_samples)) {
            return std::nullopt;
        }
        std::vector<Eigen::Vector3d> pts;
        pts.reserve(inliers.size());
        for (int idx : inliers) {
            pts.push_back(data[static_cast<std::size_t>(idx)]);
        }
        return calib::fit_plane_svd(pts);
    }

    static auto is_degenerate(const std::vector<Datum>& data, std::span<const int> sample) -> bool {
        if (sample.size() < k_min_samples) {
            return true;
        }
        const Eigen::Vector3d& p0 = data[static_cast<std::size_t>(sample[0])];
        const Eigen::Vector3d& p1 = data[static_cast<std::size_t>(sample[1])];
        const Eigen::Vector3d& p2 = data[static_cast<std::size_t>(sample[2])];
        const Eigen::Vector3d v1 = p1 - p0;
        const Eigen::Vector3d v2 = p2 - p0;
        return v1.cross(v2).norm() < 1e-12;
    }
};

}  // namespace

namespace calib {

auto fit_plane_svd(const std::vector<Eigen::Vector3d>& pts) -> Eigen::Vector4d {
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

auto fit_plane_ransac(const std::vector<Eigen::Vector3d>& pts, const RansacOptions& opts)
    -> PlaneRansacResult {
    PlaneRansacResult result;
    if (pts.size() < PlaneRansacEstimator::k_min_samples) {
        return result;
    }

    auto ransac_res = ransac<PlaneRansacEstimator>(pts, opts);
    if (!ransac_res.success) {
        return result;
    }

    result.success = true;
    result.plane = ransac_res.model;
    result.inliers = std::move(ransac_res.inliers);
    result.inlier_rms = ransac_res.inlier_rms;
    return result;
}

}  // namespace calib
