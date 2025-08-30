// Nonlinear homography refinement with Ceres
// - Input: N, then N lines: x y x' y'
// - Output: refined 3x3 homography H (last element fixed to 1)

// std
#include <iostream>

#include "calib/homography.h"

using Vec2 = Eigen::Vector2d;
using Mat3 = Eigen::Matrix3d;

using namespace calib;

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int N;
    if (!(std::cin >> N) || N < 4) {
        std::cerr << "Provide N>=4 correspondences, then N lines: x y u v\n";
        return 1;
    }

    std::vector<Vec2> src(N), dst(N);
    for (int i = 0; i < N; ++i) {
        double x, y, u, v;
        if (!(std::cin >> x >> y >> u >> v)) {
            std::cerr << "Failed to read correspondence " << i << "\n";
            return 1;
        }
        src[i] = Vec2(x, y);
        dst[i] = Vec2(u, v);
    }

    auto H = fit_homography(src, dst);

    std::cout.setf(std::ios::fixed); std::cout.precision(10);
    std::cout << "Refined H:\n";
    for (int r = 0; r < 3; ++r) {
        std::cout << H(r,0) << " " << H(r,1) << " " << H(r,2) << "\n";
    }

    // Report mean reprojection error (pixels)
    double total_err = 0.0;
    for (int i = 0; i < N; ++i) {
        Eigen::Vector3d p(src[i].x(), src[i].y(), 1.0);
        Eigen::Vector3d q = H * p; q /= q.z();
        double du = q.x() - dst[i].x();
        double dv = q.y() - dst[i].y();
        total_err += std::sqrt(du*du + dv*dv);
    }

    std::cout << "Mean reprojection error: " << (total_err / N) << " px\n";
}
