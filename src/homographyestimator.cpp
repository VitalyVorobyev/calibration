#include "homographyestimator.h"

namespace calib {

// Hartley normalization
static auto normalize(const std::vector<Eigen::Vector2d>& P, std::vector<Eigen::Vector2d>& Pn,
                      Eigen::Matrix3d& T) {
    Eigen::Vector2d c(0, 0);
    for (auto& p : P) c += p;
    c /= std::max<size_t>(1, P.size());

    double mean_dist = 0.0;
    for (auto& p : P) mean_dist += (p - c).norm();
    mean_dist /= std::max<size_t>(1, P.size());
    double s = (mean_dist > 0) ? std::sqrt(2.0) / mean_dist : 1.0;

    T.setIdentity();
    T(0, 0) = s;
    T(1, 1) = s;
    T(0, 2) = -s * c.x();
    T(1, 2) = -s * c.y();

    Pn.resize(P.size());
    for (size_t i = 0; i < P.size(); ++i) {
        Eigen::Vector3d hp(P[i].x(), P[i].y(), 1.0);
        Eigen::Vector3d hn = T * hp;
        Pn[i] = hn.hnormalized();
    }
}

static auto dlt_homography_normalized(const std::vector<Eigen::Vector2d>& src_xy,
                                                 const std::vector<Eigen::Vector2d>& dst_uv) -> Eigen::Matrix3d {
    // Build A from normalized correspondences (2N x 9)
    const int npts = static_cast<int>(src_xy.size());
    Eigen::MatrixXd amtx(2 * npts, 9);
    for (int i = 0; i < npts; ++i) {
        const double xcoord = src_xy[i].x();
        const double ycoord = src_xy[i].y();
        const double ucoord = dst_uv[i].x();
        const double vcoord = dst_uv[i].y();

        amtx.row(2 * i) << -xcoord, -ycoord, -1, 0, 0, 0, ucoord * xcoord, ucoord * ycoord, ucoord;
        amtx.row(2 * i + 1) << 0, 0, 0, -xcoord, -ycoord, -1, vcoord * xcoord, vcoord * ycoord, vcoord;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(amtx, Eigen::ComputeFullV);
    Eigen::VectorXd hvec = svd.matrixV().col(8);
    Eigen::Matrix3d hmtx;
    hmtx << hvec(0), hvec(1), hvec(2), hvec(3), hvec(4), hvec(5), hvec(6), hvec(7), hvec(8);
    return hmtx / hmtx(2, 2);
}

// --- Helpers: and normalized DLT ---
static auto normalize_and_dlt(const std::vector<Eigen::Vector2d>& src,
                              const std::vector<Eigen::Vector2d>& dst) -> Eigen::Matrix3d {
    std::vector<Eigen::Vector2d> src_n, dst_n;
    Eigen::Matrix3d t_src;
    Eigen::Matrix3d t_dst;
    normalize(src, src_n, t_src);
    normalize(dst, dst_n, t_dst);

    Eigen::Matrix3d hnorm = dlt_homography_normalized(src_n, dst_n);  // Hn: src_n -> dst_n
    return t_dst.inverse() * hnorm * t_src;
}

static double symmetric_transfer_error(const Eigen::Matrix3d& H, const Eigen::Vector2d& XY,
                                       const Eigen::Vector2d& UV) {
    auto hmul = [](const Eigen::Matrix3d& M, const Eigen::Vector2d& p) {
        Eigen::Vector3d hp(p.x(), p.y(), 1.0);
        Eigen::Vector3d q = M * hp;
        return (q / q.z()).hnormalized();
    };
    Eigen::Matrix3d Hinv = H.inverse();
    Eigen::Vector2d uv_hat = hmul(H, XY);
    Eigen::Vector2d xy_hat = hmul(Hinv, UV);
    double e1 = (UV - uv_hat).norm();
    double e2 = (XY - xy_hat).norm();
    return std::sqrt(0.5 * (e1 * e1 + e2 * e2));
}

using Model = HomographyEstimator::Model;
using Datum = HomographyEstimator::Datum;

auto HomographyEstimator::fit(const std::vector<Datum>& data, std::span<const int> sample) const -> std::optional<Model> {
    if (sample.size() < k_min_samples) return std::nullopt;
    std::vector<Eigen::Vector2d> src, dst;
    src.reserve(sample.size());
    dst.reserve(sample.size());
    for (int idx : sample) {
        src.push_back(data[idx].object_xy);
        dst.push_back(data[idx].image_uv);
    }
    Eigen::Matrix3d H = normalize_and_dlt(src, dst);
    if (!std::isfinite(H(0, 0))) return std::nullopt;
    return H;
}

auto HomographyEstimator::residual(const Model& H, const Datum& d) const -> double {
    return symmetric_transfer_error(H, d.object_xy, d.image_uv);
}

// Optional: better final model on all inliers
auto HomographyEstimator::refit(const std::vector<Datum>& data, std::span<const int> inliers) const -> std::optional<Model> {
    if (inliers.size() < k_min_samples) return std::nullopt;
    std::vector<Eigen::Vector2d> src, dst;
    src.reserve(inliers.size());
    dst.reserve(inliers.size());
    for (int idx : inliers) {
        src.push_back(data[idx].object_xy);
        dst.push_back(data[idx].image_uv);
    }
    Eigen::Matrix3d H = normalize_and_dlt(src, dst);
    if (!std::isfinite(H(0, 0))) return std::nullopt;
    return H;
}

// Optional: reject degenerate minimal sets (near-collinear points)
auto HomographyEstimator::is_degenerate(const std::vector<Datum>& data, std::span<const int> sample) const -> bool {
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
                if (a2 < eps) return true;  // near-collinear
            }
        }
    }
    return false;
}

}  // namespace calib
