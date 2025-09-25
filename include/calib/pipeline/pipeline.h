#pragma once

// std
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

// third-party
#include <nlohmann/json.hpp>

#include "calib/pipeline/dataset.h"
#include "calib/estimation/extrinsics.h"
#include "calib/pipeline/extrinsics.h"
#include "calib/pipeline/planar_intrinsics.h"

namespace calib::pipeline {

struct PipelineStageResult {
    std::string name;
    bool success{false};
    nlohmann::json summary;
};

struct PipelineExecutionReport {
    bool success{false};
    std::vector<PipelineStageResult> stages;
};

class CalibrationStage;
class StageDecorator;
class DatasetLoader;

class PipelineContext {
  public:
    CalibrationDataset dataset;
    std::unordered_map<std::string, planar::CalibrationRunResult> intrinsic_results;
    std::unordered_map<std::string, ExtrinsicOptimizationResult<PinholeCamera<BrownConradyd>>>
        stereo_results;
    nlohmann::json artifacts;

    void set_intrinsics_config(planar::PlanarCalibrationConfig cfg);
    void set_stereo_config(StereoCalibrationConfig cfg);
    [[nodiscard]] auto has_intrinsics_config() const -> bool { return has_intrinsics_config_; }
    [[nodiscard]] auto intrinsics_config() const -> const planar::PlanarCalibrationConfig& {
        return intrinsics_config_;
    }
    auto intrinsics_config() -> planar::PlanarCalibrationConfig& { return intrinsics_config_; }
    [[nodiscard]] auto has_stereo_config() const -> bool { return has_stereo_config_; }
    [[nodiscard]] auto stereo_config() const -> const StereoCalibrationConfig& {
        return stereo_config_;
    }
    auto stereo_config() -> StereoCalibrationConfig& { return stereo_config_; }

  private:
    planar::PlanarCalibrationConfig intrinsics_config_;
    bool has_intrinsics_config_{false};
    StereoCalibrationConfig stereo_config_;
    bool has_stereo_config_{false};
};

class CalibrationStage {
  public:
    virtual ~CalibrationStage() = default;
    [[nodiscard]] virtual auto name() const -> std::string = 0;
    [[nodiscard]] virtual auto run(PipelineContext& context) -> PipelineStageResult = 0;
};

class StageDecorator {
  public:
    virtual ~StageDecorator() = default;
    virtual void before_stage(const CalibrationStage& /*stage*/, PipelineContext& /*context*/) {}
    virtual void after_stage(const CalibrationStage& /*stage*/, PipelineContext& /*context*/,
                             const PipelineStageResult& /*result*/) {}
};

class DatasetLoader {
  public:
    virtual ~DatasetLoader() = default;
    [[nodiscard]] virtual auto load() -> CalibrationDataset = 0;
};

class CalibrationPipeline {
  public:
    void add_stage(std::unique_ptr<CalibrationStage> stage);
    void add_decorator(std::shared_ptr<StageDecorator> decorator);

    [[nodiscard]] auto execute(DatasetLoader& loader, PipelineContext& context)
        -> PipelineExecutionReport;

  private:
    std::vector<std::unique_ptr<CalibrationStage>> stages_;
    std::vector<std::shared_ptr<StageDecorator>> decorators_;
};

class LoggingDecorator final : public StageDecorator {
  public:
    explicit LoggingDecorator(std::ostream& out) : out_(out) {}

    void before_stage(const CalibrationStage& stage, PipelineContext& context) override;
    void after_stage(const CalibrationStage& stage, PipelineContext& context,
                     const PipelineStageResult& result) override;

  private:
    std::ostream& out_;
};

}  // namespace calib::pipeline
