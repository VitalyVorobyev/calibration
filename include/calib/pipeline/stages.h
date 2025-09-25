#pragma once

#include "calib/pipeline/pipeline.h"

namespace calib::pipeline {

class IntrinsicStage final : public CalibrationStage {
  public:
    [[nodiscard]] auto name() const -> std::string override { return "intrinsics"; }
    [[nodiscard]] auto run(PipelineContext& context) -> PipelineStageResult override;
};

class StereoCalibrationStage final : public CalibrationStage {
  public:
    [[nodiscard]] auto name() const -> std::string override { return "stereo"; }
    [[nodiscard]] auto run(PipelineContext& context) -> PipelineStageResult override;
};

class HandEyeCalibrationStage final : public CalibrationStage {
  public:
    [[nodiscard]] auto name() const -> std::string override { return "hand_eye"; }
    [[nodiscard]] auto run(PipelineContext& context) -> PipelineStageResult override;
};

}  // namespace calib::pipeline
