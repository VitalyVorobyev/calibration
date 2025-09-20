#include "homographyestimator.h"

// std
#include <algorithm>
#include <numbers>
#include <numeric>
#include <random>
#include <span>
#include <vector>

#include "calib/homography.h"  // for estimate_homography

namespace calib {

// Hartley normalization
static auto normalize_points_2d(const std::vector<Eigen::Vector2d>& pts,
                                std::vector<Eigen::Vector2d>& out) -> Eigen::Matrix3d {
    out.resize(pts.size());

    const Eigen::Vector2d centroid =
        std::accumulate(pts.begin(), pts.end(), Eigen::Vector2d{0, 0}) /
        std::max<size_t>(1, pts.size());
    const double mean_dist = std::accumulate(pts.begin(), pts.end(), 0.0,
                                             [&centroid](double sum, const Eigen::Vector2d& p) {
                                                 return sum + (p - centroid).norm();
                                             }) /
                             static_cast<double>(std::max<size_t>(1, pts.size()));
    const double sigma = (mean_dist > 0) ? std::numbers::sqrt2 / mean_dist : 1.0;

    Eigen::Matrix3d transform = Eigen::Matrix3d::Identity();
    transform(0, 0) = sigma;
    transform(1, 1) = sigma;
    transform(0, 2) = -sigma * centroid.x();
    transform(1, 2) = -sigma * centroid.y();

    std::transform(pts.begin(), pts.end(), out.begin(),
                   [&transform](const Eigen::Vector2d& pt) -> Eigen::Vector2d {
                       Eigen::Vector3d hp(pt.x(), pt.y(), 1.0);
                       Eigen::Vector3d hn = transform * hp;
                       return hn.hnormalized();
                   });
    return {transform};
}

static auto dlt_homography_normalized(const std::vector<Eigen::Vector2d>& src_xy,
                                      const std::vector<Eigen::Vector2d>& dst_uv)
    -> Eigen::Matrix3d {
    // Build A from normalized correspondences (2N x 9)
    const auto npts = static_cast<Eigen::Index>(src_xy.size());
    Eigen::MatrixXd amtx(2 * npts, 9);
    for (Eigen::Index i = 0; i < npts; ++i) {
        const double xcoord = src_xy[i].x();
        const double ycoord = src_xy[i].y();
        const double ucoord = dst_uv[i].x();
        const double vcoord = dst_uv[i].y();

        amtx.row(2 * i) << -xcoord, -ycoord, -1, 0, 0, 0, ucoord * xcoord, ucoord * ycoord, ucoord;
        amtx.row(2 * i + 1) << 0, 0, 0, -xcoord, -ycoord, -1, vcoord * xcoord, vcoord * ycoord,
            vcoord;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(amtx, Eigen::ComputeFullV);
    Eigen::VectorXd hvec = svd.matrixV().col(8);
    Eigen::Matrix3d hmtx;
    hmtx << hvec(0), hvec(1), hvec(2), hvec(3), hvec(4), hvec(5), hvec(6), hvec(7), hvec(8);
    return hmtx / hmtx(2, 2);
}

static auto normalize_and_estimate_homography(const std::vector<Eigen::Vector2d>& src,
                                              const std::vector<Eigen::Vector2d>& dst)
    -> Eigen::Matrix3d {
    std::vector<Eigen::Vector2d> src_n;
    std::vector<Eigen::Vector2d> dst_n;
    const Eigen::Matrix3d t_src = normalize_points_2d(src, src_n);
    const Eigen::Matrix3d t_dst = normalize_points_2d(dst, dst_n);
    Eigen::Matrix3d hnorm = dlt_homography_normalized(src_n, dst_n);  // Hn: src_n -> dst_n
    return t_dst.inverse() * hnorm * t_src;
}

static auto symmetric_transfer_error(const Eigen::Matrix3d& hmtx, const Eigen::Vector2d& xy,
                                     const Eigen::Vector2d& uv) -> double {
    auto hmul = [](const Eigen::Matrix3d& M, const Eigen::Vector2d& p) -> Eigen::Vector2d {
        Eigen::Vector3d hp(p.x(), p.y(), 1.0);
        Eigen::Vector3d q = M * hp;
        return Eigen::Vector2d{q.hnormalized()};
    };
    const Eigen::Matrix3d hmtxinv = hmtx.inverse();
    const Eigen::Vector2d uv_hat = hmul(hmtx, xy);
    const Eigen::Vector2d xy_hat = hmul(hmtxinv, uv);
    const double e1 = (uv - uv_hat).norm();
    const double e2 = (xy - xy_hat).norm();
    return std::sqrt(0.5 * (e1 * e1 + e2 * e2));
}

using Model = HomographyEstimator::Model;
using Datum = HomographyEstimator::Datum;

auto HomographyEstimator::fit(const std::vector<Datum>& data, std::span<const int> sample)
    -> std::optional<Model> {
    if (sample.size() < k_min_samples) {
        std::cout << "HomographyEstimator::fit: sample too small\n";
        return std::nullopt;
    }
    std::vector<Eigen::Vector2d> src;
    std::vector<Eigen::Vector2d> dst;
    src.reserve(sample.size());
    dst.reserve(sample.size());
    for (int idx : sample) {
        src.push_back(data[idx].object_xy);
        dst.push_back(data[idx].image_uv);
    }
    Eigen::Matrix3d hmtx = normalize_and_estimate_homography(src, dst);
    if (!std::isfinite(hmtx(0, 0))) {
        std::cout << "HomographyEstimator::fit: non-finite homography\n";
        return std::nullopt;
    }
    return hmtx;
}

auto HomographyEstimator::residual(const Model& hmtx, const Datum& observation) -> double {
    return symmetric_transfer_error(hmtx, observation.object_xy, observation.image_uv);
}

// Optional: better final model on all inliers
auto HomographyEstimator::refit(const std::vector<Datum>& data, std::span<const int> inliers)
    -> std::optional<Model> {
    if (inliers.size() < k_min_samples) {
        return std::nullopt;
    }
    std::vector<Eigen::Vector2d> src;
    std::vector<Eigen::Vector2d> dst;
    src.reserve(inliers.size());
    dst.reserve(inliers.size());
    for (int idx : inliers) {
        src.push_back(data[idx].object_xy);
        dst.push_back(data[idx].image_uv);
    }
    Eigen::Matrix3d hmtx = normalize_and_estimate_homography(src, dst);
    if (!std::isfinite(hmtx(0, 0))) {
        return std::nullopt;
    }
    return hmtx;
}

// Optional: reject degenerate minimal sets (near-collinear points)
auto HomographyEstimator::is_degenerate(const std::vector<Datum>& data, std::span<const int> sample)
    -> bool {
    auto tri_area2 = [](const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                        const Eigen::Vector2d& c) {
        return std::abs((b - a).x() * (c - a).y() - (b - a).y() * (c - a).x());
    };
    // Check unique triples on source side
    const double eps = 1e-6;
    for (size_t i = 0; i < sample.size(); ++i) {
        for (size_t j = i + 1; j < sample.size(); ++j) {
            for (size_t k = j + 1; k < sample.size(); ++k) {
                double a2 = tri_area2(data[sample[i]].object_xy, data[sample[j]].object_xy,
                                      data[sample[k]].object_xy);
                if (a2 < eps) {
                    return true;
                }  // near-collinear
            }
        }
    }
    return false;
}

}  // namespace calib
