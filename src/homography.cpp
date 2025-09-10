
#include "calib/homography.h"

// std
#include <array>
#include <numbers>
#include <numeric>
#include <thread>

// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "ceresutils.h"

namespace calib {

using Vec2 = Eigen::Vector2d;
using Mat3 = Eigen::Matrix3d;
using HomographyParams = std::array<double, 8>;

struct HomographyBlocks final : public ProblemParamBlocks {
    HomographyParams params;

    static auto create(const Eigen::Matrix3d& init_h) -> HomographyBlocks {
        HomographyBlocks blocks;
        blocks.params = {init_h(0, 0), init_h(0, 1), init_h(0, 2), init_h(1, 0),
                         init_h(1, 1), init_h(1, 2), init_h(2, 0), init_h(2, 1)};
        return blocks;
    }

    [[nodiscard]]
    auto get_param_blocks() const -> std::vector<ParamBlock> override {
        return {{params.data(), params.size(), 8}};
    }
};

static auto params_to_h(const HomographyParams& params) -> Mat3 {
    Mat3 mat_h;
    mat_h << params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7],
        1.0;
    return mat_h;
}

#if 0
// Normalize points (Hartley): translate to centroid, scale to mean distance sqrt(2).
static void normalize_points(const std::vector<Vec2>& points, std::vector<Vec2>& norm_points,
                             Mat3& transform) {
    norm_points.resize(points.size());
    if (points.empty()) {
        transform.setIdentity();
        return;
    }
    // Centroid
    double centroid_x = 0;
    double centroid_y = 0;
    for (const auto& pt : points) {
        centroid_x += pt.x();
        centroid_y += pt.y();
    }
    centroid_x /= static_cast<double>(points.size());
    centroid_y /= static_cast<double>(points.size());
    // Mean distance to origin
    double mean_dist = 0.0;
    for (const auto& pt : points) {
        double dx = pt.x() - centroid_x;
        double dy = pt.y() - centroid_y;
        mean_dist += std::sqrt(dx * dx + dy * dy);
    }
    mean_dist /= static_cast<double>(points.size());
    constexpr double k_eps = 1e-12;
    constexpr double k_sqrt2 = std::numbers::sqrt2;
    double scale = (mean_dist > k_eps) ? k_sqrt2 / mean_dist : 1.0;
    // Similarity transform
    transform << scale, 0, -scale * centroid_x, 0, scale, -scale * centroid_y, 0, 0, 1;
    // Apply
    for (size_t idx = 0; idx < points.size(); ++idx) {
        Eigen::Vector3d ph(points[idx].x(), points[idx].y(), 1.0);
        Eigen::Vector3d q = transform * ph;
        norm_points[idx] = Vec2(q.x() / q.z(), q.y() / q.z());
    }
}
#else

static Eigen::Matrix3d normalize_points_2d(const std::vector<Eigen::Vector2d>& pts,
                                  std::vector<Eigen::Vector2d>& out) {
    out.resize(pts.size());

    Eigen::Vector2d centroid = std::accumulate(pts.begin(), pts.end(), Eigen::Vector2d::Zero());
    centroid /= std::max<size_t>(1, pts.size());

    double mean_dist = 0.0;
    for (auto& p : pts) mean_dist += (p - centroid).norm();
    mean_dist /= std::max<size_t>(1, pts.size());

    const double sigma = (mean_dist > 0) ? std::sqrt(2.0) / mean_dist : 1.0;

    Eigen::Matrix3d transform = Eigen::Matrix3d::Identity();
    transform(0, 0) = sigma;
    transform(1, 1) = sigma;
    transform(0, 2) = -sigma * centroid.x();
    transform(1, 2) = -sigma * centroid.y();

    std::transform(pts.begin(), pts.end(), out.begin(), [&](const Eigen::Vector2d& pt) {
        Eigen::Vector3d hp(pt.x(), pt.y(), 1.0);
        Eigen::Vector3d hn = transform * hp;
        return hn.hnormalized();
    });
    return {transform};
}
#endif

#if 0
// DLT initial estimate with normalization
auto estimate_homography_dlt(const std::vector<Vec2>& src, const std::vector<Vec2>& dst) -> Mat3 {
    const int num_pts = static_cast<int>(src.size());
    std::vector<Vec2> src_norm;
    std::vector<Vec2> dst_norm;
    Mat3 transform_src;
    Mat3 transform_dst;
    normalize_points(src, src_norm, transform_src);
    normalize_points(dst, dst_norm, transform_dst);
    // Build A (2N x 9)
    Eigen::MatrixXd mat_a(2 * num_pts, 9);
    for (int idx = 0; idx < num_pts; ++idx) {
        const double x = src_norm[idx].x();
        const double y = src_norm[idx].y();
        const double u = dst_norm[idx].x();
        const double v = dst_norm[idx].y();
        const auto even_idx = static_cast<Eigen::Index>(2) * idx;
        const Eigen::Index odd_idx = even_idx + 1;
        // Row 2i
        mat_a(even_idx, 0) = 0.0;
        mat_a(even_idx, 1) = 0.0;
        mat_a(even_idx, 2) = 0.0;
        mat_a(even_idx, 3) = -x;
        mat_a(even_idx, 4) = -y;
        mat_a(even_idx, 5) = -1.0;
        mat_a(even_idx, 6) = v * x;
        mat_a(even_idx, 7) = v * y;
        mat_a(even_idx, 8) = v;
        // Row 2i+1
        mat_a(odd_idx, 0) = x;
        mat_a(odd_idx, 1) = y;
        mat_a(odd_idx, 2) = 1.0;
        mat_a(odd_idx, 3) = 0.0;
        mat_a(odd_idx, 4) = 0.0;
        mat_a(odd_idx, 5) = 0.0;
        mat_a(odd_idx, 6) = -u * x;
        mat_a(odd_idx, 7) = -u * y;
        mat_a(odd_idx, 8) = -u;
    }
    // Solve Ah = 0, h = last singular vector
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(mat_a, Eigen::ComputeFullV);
    Eigen::VectorXd vec_h = svd.matrixV().col(8);
    Mat3 mat_hn;
    mat_hn << vec_h(0), vec_h(1), vec_h(2), vec_h(3), vec_h(4), vec_h(5), vec_h(6), vec_h(7),
        vec_h(8);
    // Denormalize: H = T_dst^{-1} * Hn * T_src
    Mat3 mat_h = transform_dst.inverse() * mat_hn * transform_src;
    // Fix scale
    constexpr double k_eps = 1e-15;
    if (std::abs(mat_h(2, 2)) > k_eps) {
        mat_h /= mat_h(2, 2);
    } else {
        mat_h /= (mat_h.norm() + k_eps);
    }
    return mat_h;
}
#else
static Eigen::Matrix3d dlt_homography_normalized(const std::vector<Eigen::Vector2d>& src_xy,
                                                 const std::vector<Eigen::Vector2d>& dst_uv) {
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

static HomographyResult estimate_homography(const std::vector<Eigen::Vector2d>& src_xy,
                                            const std::vector<Eigen::Vector2d>& dst_uv) {
    HomographyResult result;
    if (src_xy.size() < 4 || dst_uv.size() != src_xy.size()) return result;

    // Hartley normalization on both sides
    std::vector<Eigen::Vector2d> n_src, n_dst;
    auto transform_src = normalize_points_2d(src_xy, n_src);
    auto transform_dst = normalize_points_2d(dst_uv, n_dst);

    // DLT
    Eigen::Matrix3d hmtx_norm = dlt_homography_normalized(n_src, n_dst);
    // Denormalize
    result.hmtx = transform_dst.inverse() * hmtx_norm * transform_src;

    // All-inlier by default; the caller may override with RANSAC
    result.success = true;
    result.inliers.resize(src_xy.size());
    std::iota(result.inliers.begin(), result.inliers.end(), 0);
    result.symmetric_rms_px = 0.0;
    return result;
}
#endif

static double point_transfer_error_px(const Eigen::Matrix3d& H, const Eigen::Vector2d& X,
                                      const Eigen::Vector2d& x) {
    // forward transfer |x - H*X|
    Eigen::Vector3d Xh(X.x(), X.y(), 1.0);
    Eigen::Vector3d xh(x.x(), x.y(), 1.0);

    Eigen::Vector3d Hx = H * Xh;
    Eigen::Vector2d x_hat = (Hx / Hx.z()).hnormalized();
    double e1 = (x - x_hat).norm();

    // backward transfer |X - H^{-1}*x|
    Eigen::Matrix3d Hinv = H.inverse();
    Eigen::Vector3d HX = Hinv * xh;
    Eigen::Vector2d X_hat = (HX / HX.z()).hnormalized();
    double e2 = (X - X_hat).norm();

    return std::sqrt(0.5 * (e1 * e1 + e2 * e2));
}

static HomographyResult ransac_homography(const std::vector<Eigen::Vector2d>& src_xy,
                                          const std::vector<Eigen::Vector2d>& dst_uv, int max_iters,
                                          double thresh_px, int min_inliers) {
    HomographyResult best;
    if (src_xy.size() < 4) return best;

    std::mt19937_64 rng{1234567};
    std::uniform_int_distribution<int> uni(0, static_cast<int>(src_xy.size()) - 1);

    auto choose4 = [&](std::array<int, 4>& idx) {
        for (;;) {
            for (int k = 0; k < 4; ++k) idx[k] = uni(rng);
            // ensure uniqueness
            std::sort(idx.begin(), idx.end());
            if (std::unique(idx.begin(), idx.end()) == idx.end()) return;
        }
    };

    for (int it = 0; it < max_iters; ++it) {
        std::array<int, 4> idx;
        choose4(idx);

        std::vector<Eigen::Vector2d> s(4), d(4);
        for (int k = 0; k < 4; ++k) {
            s[k] = src_xy[idx[k]];
            d[k] = dst_uv[idx[k]];
        }

        HomographyResult cand = estimate_homography(s, d);
        if (!cand.success) continue;

        // Score all points
        std::vector<int> inliers;
        inliers.reserve(src_xy.size());
        double sum_sq = 0.0;
        int cnt = 0;

        for (int i = 0; i < (int)src_xy.size(); ++i) {
            double e = point_transfer_error_px(cand.H, src_xy[i], dst_uv[i]);
            if (e < thresh_px) {
                inliers.push_back(i);
                sum_sq += e * e;
                ++cnt;
            }
        }
        if ((int)inliers.size() < min_inliers) continue;

        // Refit on inliers (full DLT with normalization)
        std::vector<Eigen::Vector2d> s_in, d_in;
        s_in.reserve(inliers.size());
        d_in.reserve(inliers.size());
        for (int id : inliers) {
            s_in.push_back(src_xy[id]);
            d_in.push_back(dst_uv[id]);
        }
        HomographyResult refit = estimate_homography(s_in, d_in);
        if (!refit.success) continue;

        // Compute symmetric RMS on inliers
        double ss = 0.0;
        for (int id : inliers) {
            double e = point_transfer_error_px(refit.H, src_xy[id], dst_uv[id]);
            ss += e * e;
        }
        refit.inliers = std::move(inliers);
        refit.symmetric_rms_px = std::sqrt(ss / std::max(1, (int)refit.inliers.size()));

        // Keep best (lowest RMS, then most inliers)
        bool better = false;
        if (!best.success)
            better = true;
        else if (refit.inliers.size() > best.inliers.size())
            better = true;
        else if (refit.inliers.size() == best.inliers.size() &&
                 refit.symmetric_rms_px < best.symmetric_rms_px)
            better = true;

        if (better) best = std::move(refit);
    }

    if (!best.success) {
        // fallback to all-inlier DLT (no RANSAC)
        best = estimate_homography(src_xy, dst_uv);
        if (best.success) {
            double ss = 0.0;
            for (int i = 0; i < (int)src_xy.size(); ++i) {
                double e = point_transfer_error_px(best.H, src_xy[i], dst_uv[i]);
                ss += e * e;
            }
            best.symmetric_rms_px = std::sqrt(ss / std::max(1, (int)src_xy.size()));
        }
    }
    return best;
}

// Ceres residual: maps (x,y) -> (u,v) using H(h) and compares to (u*, v*)
struct HomographyResidual {
    double x_, y_, u_, v_;

    HomographyResidual(double x, double y, double u, double v) : x_(x), y_(y), u_(u), v_(v) {}

    template <typename T>
    bool operator()(const T* const h, T* residuals) const {
        // h[0..7]: 8 params, H22 = 1
        // 0 1 2
        // 3 4 5
        // 6 7 8
        constexpr size_t h00idx = 0;
        constexpr size_t h01idx = 1;
        constexpr size_t h02idx = 2;
        constexpr size_t h10idx = 3;
        constexpr size_t h11idx = 4;
        constexpr size_t h12idx = 5;
        constexpr size_t h30idx = 6;
        constexpr size_t h31idx = 7;
        T xvar = T(x_);
        T yvar = T(y_);
        T den = h[h30idx] * xvar + h[h31idx] * yvar + T(1);
        // Guard against division by zero in autodiff context
        if (ceres::isnan(den) || ceres::abs(den) < T(1e-15)) {
            residuals[0] = T(0);
            residuals[1] = T(0);
            return true;
        }
        T uu = (h[h00idx] * xvar + h[h01idx] * yvar + h[h02idx]) / den;
        T vv = (h[h10idx] * xvar + h[h11idx] * yvar + h[h12idx]) / den;
        residuals[0] = uu - T(u_);
        residuals[1] = vv - T(v_);
        return true;
    }

    static ceres::CostFunction* create(double x, double y, double u, double v) {
        return new ceres::AutoDiffCostFunction<HomographyResidual, 2, 8>(
            new HomographyResidual(x, y, u, v));
    }
};

static ceres::Problem build_problem(const std::vector<Vec2>& src, const std::vector<Vec2>& dst,
                                    const HomographyOptions& options, HomographyBlocks& blocks) {
    ceres::Problem problem;
    for (size_t i = 0; i < src.size(); ++i) {
        problem.AddResidualBlock(
            HomographyResidual::create(src[i].x(), src[i].y(), dst[i].x(), dst[i].y()),
            options.huber_delta > 0 ? new ceres::HuberLoss(options.huber_delta) : nullptr,
            blocks.params.data());
    }
    return problem;
}

OptimizeHomographyResult optimize_homography(const std::vector<Vec2>& src,
                                             const std::vector<Vec2>& dst,
                                             const Eigen::Matrix3d& init_h,
                                             const HomographyOptions& options) {
    if (src.size() < 4 || src.size() != dst.size()) {
        throw std::invalid_argument("At least 4 correspondences are required.");
    }

    auto blocks = HomographyBlocks::create(init_h);
    ceres::Problem problem = build_problem(src, dst, options, blocks);

    OptimizeHomographyResult result;
    solve_problem(problem, options, &result);

    Mat3 hmtx = params_to_h(blocks.params);
    if (std::abs(hmtx(2, 2)) > 1e-15) {
        hmtx /= hmtx(2, 2);
    }
    result.homography = hmtx;

    if (options.compute_covariance) {
        std::vector<double> residuals(src.size() * 2);
        ceres::Problem::EvaluateOptions evopts;
        problem.Evaluate(evopts, nullptr, &residuals, nullptr, nullptr);
        const double ssr = std::accumulate(residuals.begin(), residuals.end(), 0.0,
                                           [](double sum, double r) { return sum + r * r; });
        auto optcov = compute_covariance(blocks, problem, ssr, residuals.size());
        if (optcov.has_value()) {
            result.covariance = std::move(optcov.value());
        }
    }

    return result;
}

}  // namespace calib
