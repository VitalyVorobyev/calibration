/** @brief Classical Zhang camera calibration method */

#pragma once

#include <optional>
#include <vector>

#include "calib/estimation/linear/homography.h"
#include "calib/models/camera_matrix.h"

namespace calib {

auto zhang_intrinsics_from_hs(const std::vector<HomographyResult>& hs)
    -> std::optional<CameraMatrix>;

}  // namespace calib
