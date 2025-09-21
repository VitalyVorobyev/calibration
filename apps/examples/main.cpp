// Nonlinear homography refinement with Ceres
// - Input: N, then N lines: x y x' y'
// - Output: refined 3x3 homography H (last element fixed to 1)

// std
#include <iostream>

#include "calib/estimation/homography.h"

using Vec2 = Eigen::Vector2d;
using Mat3 = Eigen::Matrix3d;

using namespace calib;

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    size_t N;
    if (!(std::cin >> N) || N < 4) {
        std::cerr << "Provide N>=4 correspondences, then N lines: x y u v\n";
        return 1;
    }

    PlanarView view(N);
    for (size_t i = 0; i < N; ++i) {
        double x, y, u, v;
        if (!(std::cin >> x >> y >> u >> v)) {
            std::cerr << "Failed to read correspondence " << i << "\n";
            return 1;
        }

        view[i].image_uv = Vec2(u, v);
        view[i].object_xy = Vec2(x, y);
    }

    auto hres = estimate_homography(view);
    if (!hres.success) {
        std::cerr << "Failed to estimate homography\n";
        return 1;
    }

    std::cout.setf(std::ios::fixed);
    std::cout.precision(10);
    std::cout << "Refined H:\n";
    for (int r = 0; r < 3; ++r) {
        std::cout << hres.hmtx(r, 0) << " " << hres.hmtx(r, 1) << " " << hres.hmtx(r, 2) << "\n";
    }

    std::cout << "Mean reprojection error: " << hres.symmetric_rms_px << " px\n";
}
