#include <gtest/gtest.h>

#include <random>

#include "calib/estimation/linear/planefit.h"

using namespace calib;

namespace {

Eigen::Vector4d make_plane_from_point_normal(const Eigen::Vector3d& point,
                                             const Eigen::Vector3d& normal) {
    Eigen::Vector3d n = normal.normalized();
    double d = -n.dot(point);
    return {n.x(), n.y(), n.z(), d};
}

Eigen::Vector4d align(const Eigen::Vector4d& plane, const Eigen::Vector4d& reference) {
    return (plane.head<3>().dot(reference.head<3>()) < 0.0) ? -plane : plane;
}

}  // namespace

TEST(PlaneFit, RansacRejectsOutliers) {
    std::mt19937 rng(1337);
    std::uniform_real_distribution<double> dist_xy(-1.0, 1.0);
    Eigen::Vector3d normal(0.2, -0.3, 1.0);
    Eigen::Vector4d ground_truth = make_plane_from_point_normal({0.0, 0.0, 1.0}, normal);

    std::vector<Eigen::Vector3d> points;
    points.reserve(150);

    const int inliers = 100;
    for (int i = 0; i < inliers; ++i) {
        const double x = dist_xy(rng);
        const double y = dist_xy(rng);
        Eigen::Vector3d p(x, y, 0.0);
        p.z() = (-ground_truth[3] - ground_truth[0] * x - ground_truth[1] * y) / ground_truth[2];
        points.push_back(p);
    }

    const int outliers = 40;
    std::uniform_real_distribution<double> dist_out(5.0, 10.0);
    for (int i = 0; i < outliers; ++i) {
        points.emplace_back(dist_out(rng), dist_out(rng), dist_out(rng));
    }

    RansacOptions opts;
    opts.max_iters = 2000;
    opts.thresh = 0.01;
    opts.min_inliers = 80;
    opts.confidence = 0.999;

    auto result = fit_plane_ransac(points, opts);
    ASSERT_TRUE(result.success);
    EXPECT_GE(result.inliers.size(), static_cast<std::size_t>(inliers));

    Eigen::Vector4d estimated = align(result.plane, ground_truth);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(estimated[i], ground_truth[i], 1e-3);
    }

    EXPECT_LT(result.inlier_rms, 1e-3);

    std::size_t counted_inliers = 0;
    for (int idx : result.inliers) {
        const auto& p = points[static_cast<std::size_t>(idx)];
        const double residual = std::abs(result.plane.head<3>().dot(p) + result.plane[3]);
        if (residual < opts.thresh) {
            ++counted_inliers;
        }
    }
    EXPECT_EQ(counted_inliers, result.inliers.size());
}

TEST(PlaneFit, SvdMatchesIdealPlane) {
    Eigen::Vector3d normal(0.4, 0.1, 1.0);
    Eigen::Vector4d gt = make_plane_from_point_normal({0.5, -0.2, 0.8}, normal);

    std::vector<Eigen::Vector3d> pts;
    pts.reserve(50);
    for (int i = -5; i <= 5; ++i) {
        for (int j = -5; j <= 5; ++j) {
            Eigen::Vector3d p(static_cast<double>(i) * 0.1, static_cast<double>(j) * 0.1, 0.0);
            p.z() = (-gt[3] - gt[0] * p.x() - gt[1] * p.y()) / gt[2];
            pts.push_back(p);
        }
    }

    Eigen::Vector4d estimated = fit_plane_svd(pts);
    estimated = align(estimated, gt);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(estimated[i], gt[i], 1e-9);
    }
}
