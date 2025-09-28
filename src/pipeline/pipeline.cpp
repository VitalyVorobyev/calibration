#include "calib/pipeline/pipeline.h"

// std
#include <utility>

namespace calib::pipeline {

void PipelineContext::set_intrinsics_config(planar::PlanarCalibrationConfig cfg) {
    intrinsics_config_ = std::move(cfg);
    has_intrinsics_config_ = true;
}

void PipelineContext::set_stereo_config(StereoCalibrationConfig cfg) {
    stereo_config_ = std::move(cfg);
    has_stereo_config_ = true;
}

void PipelineContext::set_handeye_config(HandEyePipelineConfig cfg) {
    handeye_config_ = std::move(cfg);
    has_handeye_config_ = true;
}

void PipelineContext::set_bundle_config(BundlePipelineConfig cfg) {
    bundle_config_ = std::move(cfg);
    has_bundle_config_ = true;
}

void CalibrationPipeline::add_stage(std::unique_ptr<CalibrationStage> stage) {
    stages_.push_back(std::move(stage));
}

void CalibrationPipeline::add_decorator(std::shared_ptr<StageDecorator> decorator) {
    decorators_.push_back(std::move(decorator));
}

auto CalibrationPipeline::execute(DatasetLoader& loader, PipelineContext& context)
    -> PipelineExecutionReport {
    context.dataset = loader.load();

    PipelineExecutionReport report;
    report.success = true;

    for (const auto& stage : stages_) {
        for (const auto& decorator : decorators_) {
            decorator->before_stage(*stage, context);
        }

        auto stage_result = stage->run(context);
        if (stage_result.name.empty()) {
            stage_result.name = stage->name();
        }

        for (const auto& decorator : decorators_) {
            decorator->after_stage(*stage, context, stage_result);
        }

        report.success = report.success && stage_result.success;
        report.stages.push_back(std::move(stage_result));
    }

    return report;
}

void LoggingDecorator::before_stage(const CalibrationStage& stage, PipelineContext&) {
    out_ << "[pipeline] → Starting stage '" << stage.name() << "'" << '\n';
}

void LoggingDecorator::after_stage(const CalibrationStage& stage, PipelineContext&,
                                   const PipelineStageResult& result) {
    out_ << "[pipeline] ← Completed stage '" << stage.name() << "'"
         << (result.success ? " (success)" : " (failed)") << '\n';
}

}  // namespace calib::pipeline
