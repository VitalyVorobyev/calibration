// Nonlinear homography refinement with Ceres
// - Input: N, then N lines: x y x' y'
// - Output: refined 3x3 homography H (last element fixed to 1)

// std
#include <iostream>
#include <spdlog/spdlog.h>

#include "calib/homography.h"

using Vec2 = Eigen::Vector2d;
using Mat3 = Eigen::Matrix3d;

using namespace calib;

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    size_t N;
    if (!(std::cin >> N) || N < 4) {
        spdlog::error("Provide N>=4 correspondences, then N lines: x y u v");
        return 1;
    }

    PlanarView view(N);
    for (int i = 0; i < N; ++i) {
        double x, y, u, v;
        if (!(std::cin >> x >> y >> u >> v)) {
            spdlog::error("Failed to read correspondence {}", i);
            return 1;
        }

        view[i].image_uv = Vec2(u, v);
        view[i].object_xy = Vec2(x, y);
    }

    auto hres = estimate_homography(view);
    if (!hres.success) {
        spdlog::error("Failed to estimate homography");
        return 1;
    }
    spdlog::info("Refined H:");
    for (int r = 0; r < 3; ++r) {
        spdlog::info("{:.10f} {:.10f} {:.10f}", hres.hmtx(r,0), hres.hmtx(r,1), hres.hmtx(r,2));
    }

    spdlog::info("Mean reprojection error: {:.10f} px", hres.symmetric_rms_px);
}
