/** @brief Classical Zhang camera calibration method */

#pragma once

#include "calib/core/cameramatrix.h"
#include "calib/estimation/homography.h"

namespace calib {

auto zhang_intrinsics_from_hs(const std::vector<HomographyResult>& hs)
    -> std::optional<CameraMatrix>;

}  // namespace calib
