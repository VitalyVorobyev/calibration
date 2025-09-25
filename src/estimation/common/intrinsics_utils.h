#pragma once

// std
#include <optional>
#include <utility>

#include "calib/models/camera_matrix.h"

namespace calib::detail {

[[nodiscard]] auto sanitize_intrinsics(const CameraMatrix& kmtx,
                                       const std::optional<CalibrationBounds>& bounds)
    -> std::pair<CameraMatrix, bool>;

}  // namespace calib::detail
