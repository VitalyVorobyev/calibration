/** @brief Classical Zhang camera calibration method */

#pragma once

#include "calib/cameramatrix.h"
#include "calib/homography.h"

namespace calib {

auto zhang_intrinsics_from_hs(const std::vector<HomographyResult>& hs)
    -> std::optional<CameraMatrix>;

}  // namespace calib
