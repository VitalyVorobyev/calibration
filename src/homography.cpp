
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
#include "homographyestimator.h"

namespace calib {

static auto symmetric_rms_px(const Eigen::Matrix3d& hmtx,
                             const std::vector<PlanarObservation>& data,
                             std::span<const int> inliers) -> double {
    if (inliers.size() == 0) return std::numeric_limits<double>::infinity();
    HomographyEstimator estimator;
    const double sum_sq_err = std::accumulate(
        inliers.begin(), inliers.end(), 0.0,
        [&](double acc, int idx) { return acc + estimator.residual(hmtx, data[idx]); });
    return std::sqrt(sum_sq_err / (2.0 * inliers.size()));
}

auto estimate_homography_dlt(const std::vector<PlanarObservation>& data,
                             std::optional<RansacOptions> ransac_opts) -> HomographyResult {
    HomographyResult result;
    HomographyEstimator estimator;

    if (!ransac_opts.has_value()) {
        auto hmtx_opt = estimator.fit(data, std::span<const int>());
        if (!hmtx_opt.has_value()) {
            std::cout << "Homography estimation failed.\n";
            return result;
        }
        result.hmtx = hmtx_opt.value();
        result.symmetric_rms_px = symmetric_rms_px(result.hmtx, data, std::span<const int>());
        result.inliers.resize(data.size());
        std::iota(result.inliers.begin(), result.inliers.end(), 0);
    } else {
        auto ransac_result = ransac(data, estimator, ransac_opts.value());
        if (!ransac_result.success) {
            std::cout << "Homography RANSAC failed.\n";
            return result;
        }
        result.hmtx = ransac_result.model;
        result.inliers = std::move(ransac_result.inliers);
        result.symmetric_rms_px = symmetric_rms_px(result.hmtx, data, result.inliers);
    }

    std::cout << "Homography inliers: " << result.inliers.size() << " / " << data.size()
              << ", symmetric RMS: " << result.symmetric_rms_px << " px\n";

    result.success = true;
    return result;
}

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
        Eigen::Matrix<T, 3, 3> hmtx;
        hmtx << h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], T(1);
        T xvar = ;
        T yvar = T(y_);
        Eigen::Vector3<T> xyvec(T(x_), T(y_), T(1));

        #if 0
        // Guard against division by zero in autodiff context
        if (ceres::isnan(den) || ceres::abs(den) < T(1e-15)) {
            residuals[0] = T(0);
            residuals[1] = T(0);
            return true;
        }
        #endif

        Eigen::Vector2<T> uv_hat = (hmtx * xyvec).hnormalized().head<2>();
        residuals[0] = uv_hat.x() - T(u_);
        residuals[1] = uv_hat.y() - T(v_);
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
