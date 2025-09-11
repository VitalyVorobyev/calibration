#include "homographyestimator.h"

namespace calib {

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

// --- Helpers: Hartley normalization and normalized DLT ---
static auto normalize_and_dlt(const std::vector<Eigen::Vector2d>& src,
                              const std::vector<Eigen::Vector2d>& dst) -> Eigen::Matrix3d {
    std::vector<Eigen::Vector2d> src_n, dst_n;
    Eigen::Matrix3d Ts, Td;
    normalize(src, src_n, Ts);
    normalize(dst, dst_n, Td);

    const int N = static_cast<int>(src_n.size());
    Eigen::MatrixXd A(2 * N, 9);
    for (int i = 0; i < N; ++i) {
        double X = src_n[i].x(), Y = src_n[i].y();
        double u = dst_n[i].x(), v = dst_n[i].y();
        A.row(2 * i) << -X, -Y, -1, 0, 0, 0, u * X, u * Y, u;
        A.row(2 * i + 1) << 0, 0, 0, -X, -Y, -1, v * X, v * Y, v;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd h = svd.matrixV().col(8);

    Eigen::Matrix3d Hn;
    Hn << h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), h(8);
    Hn /= Hn(2, 2);
    // denormalize
    return Td.inverse() * Hn * Ts;
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

std::optional<Model> fit(const std::vector<Datum>& data, std::span<const int> sample) const {
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

double residual(const Model& H, const Datum& d) const {
    return symmetric_transfer_error(H, d.object_xy, d.image_uv);
}

// Optional: better final model on all inliers
std::optional<Model> refit(const std::vector<Datum>& data, std::span<const int> inliers) const {
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
bool is_degenerate(const std::vector<Datum>& data, std::span<const int> sample) const {
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
