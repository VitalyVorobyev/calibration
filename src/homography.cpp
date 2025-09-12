
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

static void estimate_homography_dlt(const PlanarView& data, HomographyResult& result,
                                    HomographyEstimator& estimator) {
    std::vector<int> inliers(data.size());
    std::iota(inliers.begin(), inliers.end(), 0);
    auto hmtx_opt = estimator.fit(data, inliers);
    if (!hmtx_opt.has_value()) {
        std::cout << "Homography estimation failed.\n";
        result.success = false;
        return;
    }
    result.hmtx = hmtx_opt.value();
    result.symmetric_rms_px = symmetric_rms_px(result.hmtx, data, inliers);
    result.inliers = inliers;
    result.success = true;
}

static void estimate_homography_ransac(const PlanarView& data, HomographyResult& result,
                                       HomographyEstimator& estimator,
                                       const RansacOptions& ransac_opts) {
    auto ransac_result = ransac(data, estimator, ransac_opts);
    if (!ransac_result.success) {
        std::cout << "Homography RANSAC failed.\n";
        result.success = false;
        return;
    }
    result.hmtx = ransac_result.model;
    result.inliers = std::move(ransac_result.inliers);
    result.symmetric_rms_px = symmetric_rms_px(result.hmtx, data, result.inliers);
    result.success = true;

    std::cout << "Homography inliers: " << result.inliers.size() << " / " << data.size()
              << ", symmetric RMS: " << result.symmetric_rms_px << " px\n";
}

auto estimate_homography(const PlanarView& data,
                         std::optional<RansacOptions> ransac_opts) -> HomographyResult {
    HomographyResult result;
    HomographyEstimator estimator;

    if (!ransac_opts.has_value()) {
        estimate_homography_dlt(data, result, estimator);
    } else {
        estimate_homography_ransac(data, result, estimator, ransac_opts.value());
    }

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
struct HomographyResidual final {
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
        Eigen::Vector3<T> xyvec(T(x_), T(y_), T(1));

#if 0
        // Guard against division by zero in autodiff context
        if (ceres::isnan(den) || ceres::abs(den) < T(1e-15)) {
            residuals[0] = T(0);
            residuals[1] = T(0);
            return true;
        }
#endif

        Eigen::Vector3<T> uvw = hmtx * xyvec;
        Eigen::Vector2<T> uv_hat = uvw.hnormalized();
        residuals[0] = uv_hat.x() - T(u_);
        residuals[1] = uv_hat.y() - T(v_);
        return true;
    }

    static ceres::CostFunction* create(double x, double y, double u, double v) {
        return new ceres::AutoDiffCostFunction<HomographyResidual, 2, 8>(
            new HomographyResidual(x, y, u, v));
    }
};

static ceres::Problem build_problem(const PlanarView& data, const HomographyOptions& options,
                                    HomographyBlocks& blocks) {
    ceres::Problem problem;
    std::for_each(data.begin(), data.end(), [&](const auto& obs) {
        problem.AddResidualBlock(
            HomographyResidual::create(obs.object_xy.x(), obs.object_xy.y(), obs.image_uv.x(),
                                       obs.image_uv.y()),
            options.huber_delta > 0 ? new ceres::HuberLoss(options.huber_delta) : nullptr,
            blocks.params.data());
    });
    return problem;
}

OptimizeHomographyResult optimize_homography(const PlanarView& data, const Eigen::Matrix3d& init_h,
                                             const HomographyOptions& options) {
    if (data.size() < 4) {
        throw std::invalid_argument("At least 4 correspondences are required.");
    }

    auto blocks = HomographyBlocks::create(init_h);
    ceres::Problem problem = build_problem(data, options, blocks);

    OptimizeHomographyResult result;
    solve_problem(problem, options, &result);

    Mat3 hmtx = params_to_h(blocks.params);
    if (std::abs(hmtx(2, 2)) > 1e-15) {
        hmtx /= hmtx(2, 2);
    }
    result.homography = hmtx;

    if (options.compute_covariance) {
        std::vector<double> residuals(data.size() * 2);
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
