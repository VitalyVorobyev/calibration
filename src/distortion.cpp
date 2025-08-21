#include "calibration/distortion.h"

namespace vitavision {

Eigen::Vector2d apply_distortion(
    const Eigen::Vector2d& norm_xy,
    const Eigen::VectorXd& coeffs
) {
    if (coeffs.size() < 2) {
        throw std::runtime_error("Insufficient distortion coefficients");
    }

    const double x = norm_xy.x();
    const double y = norm_xy.y();
    const double r2 = x*x + y*y;

    const int num_k_radial = coeffs.size() - 2;
    const double p1 = coeffs[num_k_radial];
    const double p2 = coeffs[num_k_radial + 1];

    // Apply radial distortion
    double radial = 1.0;
    double rpow = r2;

    for (int i = 0; i < num_k_radial; ++i) {
        radial += coeffs[i] * rpow;
        rpow *= r2;
    }

    // Apply tangential distortion
    double x_t = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
    double y_t = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

    return { x_t, y_t };
}

}  // namespace vitavision
