#pragma once

#include <algorithm>
#include <cmath>
#include <optional>
#include <utility>

#include "calib/models/camera_matrix.h"

namespace calib::detail {

inline auto sanitize_intrinsics_impl(const CameraMatrix& kmtx,
                                     const std::optional<CalibrationBounds>& bounds)
    -> std::pair<CameraMatrix, bool> {
    if (!bounds.has_value()) {
        return {kmtx, false};
    }

    const auto& b = bounds.value();
    CameraMatrix adjusted = kmtx;
    bool modified = false;

    const auto enforce_min_focal = [&modified](double value, double min_val) {
        if (!std::isfinite(value) || value < min_val) {
            modified = true;
            return min_val;
        }
        return value;
    };

    const auto midpoint = [](double min_val, double max_val) { return 0.5 * (min_val + max_val); };

    const auto adjust_principal_point = [&modified, &midpoint](double value, double min_val,
                                                               double max_val) {
        if (!std::isfinite(value)) {
            modified = true;
            return midpoint(min_val, max_val);
        }
        if (value < min_val || value > max_val) {
            modified = true;
            return midpoint(min_val, max_val);
        }
        return value;
    };

    adjusted.fx = enforce_min_focal(adjusted.fx, b.fx_min);
    adjusted.fy = enforce_min_focal(adjusted.fy, b.fy_min);
    adjusted.cx = adjust_principal_point(adjusted.cx, b.cx_min, b.cx_max);
    adjusted.cy = adjust_principal_point(adjusted.cy, b.cy_min, b.cy_max);

    const double skew_min = std::min(b.skew_min, b.skew_max);
    const double skew_max = std::max(b.skew_min, b.skew_max);
    if (!std::isfinite(adjusted.skew) || adjusted.skew < skew_min || adjusted.skew > skew_max) {
        modified = true;
        adjusted.skew = std::clamp(0.0, skew_min, skew_max);
    }
    return {adjusted, modified};
}

}  // namespace calib::detail

namespace calib {

inline auto sanitize_intrinsics(const CameraMatrix& kmtx,
                                const std::optional<CalibrationBounds>& bounds)
    -> std::pair<CameraMatrix, bool> {
    return detail::sanitize_intrinsics_impl(kmtx, bounds);
}

}  // namespace calib
