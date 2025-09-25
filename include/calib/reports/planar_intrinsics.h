#pragma once

#include <filesystem>
#include <nlohmann/json.hpp>

#include "calib/pipeline/planar_intrinsics.h"
#include "calib/reports/planar_intrinsics_types.h"

namespace calib::planar {

[[nodiscard]] auto build_planar_intrinsics_report(const PlanarCalibrationConfig& cfg,
                                                  const CameraConfig& cam_cfg,
                                                  const PlanarDetections& detections,
                                                  const CalibrationOutputs& outputs,
                                                  const std::filesystem::path& features_path)
    -> PlanarIntrinsicsReport;

}  // namespace calib::planar
