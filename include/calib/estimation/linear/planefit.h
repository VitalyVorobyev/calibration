#pragma once

#include <Eigen/Core>
#include <vector>

#include "calib/models/cameramodel.h"

namespace calib {

auto fit_plane_svd(const std::vector<Eigen::Vector3d>& pts) -> Eigen::Vector4d;

}  // namespace calib
