
#include "calib/homography.h"

// std
#include <array>
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
    auto get_param_blocks() const -> std::vector<ParamBlock> override { return {{params.data(), params.size(), 8}}; }
};

static auto params_to_h(const HomographyParams& params) -> Mat3 {
    Mat3 mat_h;
    mat_h << params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], 1.0;
    return mat_h;
}

// Normalize points (Hartley): translate to centroid, scale to mean distance sqrt(2).
static void normalize_points(const std::vector<Vec2>& points, std::vector<Vec2>& norm_points, Mat3& transform) {
    norm_points.resize(points.size());
    if (points.empty()) {
        transform.setIdentity();
        return;
    }
    // Centroid
    double centroid_x = 0, centroid_y = 0;
    for (const auto& pt : points) {
        centroid_x += pt.x();
        centroid_y += pt.y();
    }
    centroid_x /= static_cast<double>(points.size());
    centroid_y /= static_cast<double>(points.size());
    // Mean distance to origin
    double mean_dist = 0.0;
    for (const auto& pt : points) {
        double dx = pt.x() - centroid_x, dy = pt.y() - centroid_y;
        mean_dist += std::sqrt(dx * dx + dy * dy);
    }
    mean_dist /= static_cast<double>(points.size());
    constexpr double kEps = 1e-12;
    constexpr double kSqrt2 = std::numbers::sqrt2;
    double scale = (mean_dist > kEps) ? kSqrt2 / mean_dist : 1.0;
    // Similarity transform
    transform << scale, 0, -scale * centroid_x, 0, scale, -scale * centroid_y, 0, 0, 1;
    // Apply
    for (size_t idx = 0; idx < points.size(); ++idx) {
        Eigen::Vector3d ph(points[idx].x(), points[idx].y(), 1.0);
        Eigen::Vector3d q = transform * ph;
        norm_points[idx] = Vec2(q.x() / q.z(), q.y() / q.z());
    }
}

// DLT initial estimate with normalization
auto estimate_homography_dlt(const std::vector<Vec2>& src, const std::vector<Vec2>& dst) -> Mat3 {
    const int num_pts = static_cast<int>(src.size());
    std::vector<Vec2> src_norm, dst_norm;
    Mat3 transform_src, transform_dst;
    normalize_points(src, src_norm, transform_src);
    normalize_points(dst, dst_norm, transform_dst);
    // Build A (2N x 9)
    Eigen::MatrixXd mat_A(2 * num_pts, 9);
    for (int idx = 0; idx < num_pts; ++idx) {
        const double x = src_norm[idx].x();
        const double y = src_norm[idx].y();
        const double u = dst_norm[idx].x();
        const double v = dst_norm[idx].y();
        // Row 2i
        mat_A(static_cast<Eigen::Index>(2 * idx), 0) = 0.0;
        mat_A(static_cast<Eigen::Index>(2 * idx), 1) = 0.0;
        mat_A(static_cast<Eigen::Index>(2 * idx), 2) = 0.0;
        mat_A(static_cast<Eigen::Index>(2 * idx), 3) = -x;
        mat_A(static_cast<Eigen::Index>(2 * idx), 4) = -y;
        mat_A(static_cast<Eigen::Index>(2 * idx), 5) = -1.0;
        mat_A(static_cast<Eigen::Index>(2 * idx), 6) = v * x;
        mat_A(static_cast<Eigen::Index>(2 * idx), 7) = v * y;
        mat_A(static_cast<Eigen::Index>(2 * idx), 8) = v;
        // Row 2i+1
        mat_A(static_cast<Eigen::Index>(2 * idx + 1), 0) = x;
        mat_A(static_cast<Eigen::Index>(2 * idx + 1), 1) = y;
        mat_A(static_cast<Eigen::Index>(2 * idx + 1), 2) = 1.0;
        mat_A(static_cast<Eigen::Index>(2 * idx + 1), 3) = 0.0;
        mat_A(static_cast<Eigen::Index>(2 * idx + 1), 4) = 0.0;
        mat_A(static_cast<Eigen::Index>(2 * idx + 1), 5) = 0.0;
        mat_A(static_cast<Eigen::Index>(2 * idx + 1), 6) = -u * x;
        mat_A(static_cast<Eigen::Index>(2 * idx + 1), 7) = -u * y;
        mat_A(static_cast<Eigen::Index>(2 * idx + 1), 8) = -u;
    }
    // Solve Ah = 0, h = last singular vector
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(mat_A, Eigen::ComputeFullV);
    Eigen::VectorXd vec_h = svd.matrixV().col(8);
    Mat3 mat_Hn;
    mat_Hn << vec_h(0), vec_h(1), vec_h(2), vec_h(3), vec_h(4), vec_h(5), vec_h(6), vec_h(7), vec_h(8);
    // Denormalize: H = T_dst^{-1} * Hn * T_src
    Mat3 mat_H = transform_dst.inverse() * mat_Hn * transform_src;
    // Fix scale
    constexpr double kEps = 1e-15;
    if (std::abs(mat_H(2, 2)) > kEps) {
        mat_H /= mat_H(2, 2);
    } else {
        mat_H /= (mat_H.norm() + kEps);
    }
    return mat_H;
}

// Ceres residual: maps (x,y) -> (u,v) using H(h) and compares to (u*, v*)
struct HomographyResidual {
    double x_, y_, u_, v_;

    HomographyResidual(double x, double y, double u, double v) : x_(x), y_(y), u_(u), v_(v) {}

    template <typename T>
    bool operator()(const T* const h, T* residuals) const {
        // h[0..7]: 8 params, H22 = 1
        T X = T(x_), Y = T(y_);
        T den = h[6] * X + h[7] * Y + T(1);
        // Guard against division by zero in autodiff context
        if (ceres::isnan(den) || ceres::abs(den) < T(1e-15)) {
            residuals[0] = T(0);
            residuals[1] = T(0);
            return true;
        }
        T uu = (h[0] * X + h[1] * Y + h[2]) / den;
        T vv = (h[3] * X + h[4] * Y + h[5]) / den;
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

    Mat3 H = params_to_h(blocks.params);
    if (std::abs(H(2, 2)) > 1e-15) H /= H(2, 2);
    result.homography = H;

    if (options.compute_covariance) {
        std::vector<double> residuals(src.size() * 2);
        ceres::Problem::EvaluateOptions evopts;
        problem.Evaluate(evopts, nullptr, &residuals, nullptr, nullptr);
        double ssr = 0.0;
        for (double r : residuals) ssr += r * r;
        auto optcov = compute_covariance(blocks, problem, residuals.size(), ssr);
        if (optcov.has_value()) {
            result.covariance = std::move(optcov.value());
        }
    }

    return result;
}

}  // namespace calib
