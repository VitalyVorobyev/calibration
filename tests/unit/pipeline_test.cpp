#include "calib/pipeline/pipeline.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <sstream>

namespace calib::pipeline {
namespace {

class MockStage : public CalibrationStage {
  public:
    MOCK_METHOD(std::string, name, (), (const, override));
    MOCK_METHOD(PipelineStageResult, run, (PipelineContext & context), (override));
};

class MockDecorator : public StageDecorator {
  public:
    MOCK_METHOD(void, before_stage, (const CalibrationStage& stage, PipelineContext& context),
                (override));
    MOCK_METHOD(void, after_stage,
                (const CalibrationStage& stage, PipelineContext& context,
                 const PipelineStageResult& result),
                (override));
};

class MockDatasetLoader : public DatasetLoader {
  public:
    MOCK_METHOD(CalibrationDataset, load, (), (override));
};

TEST(CalibrationPipelineTest, AddStageEnqueuesStageForExecution) {
    PipelineContext context;
    ::testing::StrictMock<MockDatasetLoader> loader;

    CalibrationDataset dataset;
    dataset.schema_version = 42;
    dataset.metadata["source"] = "unit_test";
    EXPECT_CALL(loader, load()).WillOnce(::testing::Return(dataset));

    CalibrationPipeline pipeline;
    auto stage = std::make_unique<::testing::StrictMock<MockStage>>();
    auto* stage_ptr = stage.get();
    pipeline.add_stage(std::move(stage));

    EXPECT_CALL(*stage_ptr, name()).WillRepeatedly(::testing::Return("mock-stage"));
    EXPECT_CALL(*stage_ptr, run(::testing::Ref(context))).WillOnce([](PipelineContext&) {
        PipelineStageResult result;
        result.name = "custom-stage";
        result.success = true;
        result.summary["status"] = "ran";
        return result;
    });

    const auto report = pipeline.execute(loader, context);

    EXPECT_TRUE(report.success);
    ASSERT_EQ(report.stages.size(), 1);
    EXPECT_EQ(report.stages.front().name, "custom-stage");
    EXPECT_EQ(report.stages.front().summary.at("status").get<std::string>(), "ran");
    EXPECT_EQ(context.dataset.schema_version, 42);
    EXPECT_EQ(context.dataset.metadata.at("source").get<std::string>(), "unit_test");
}

TEST(CalibrationPipelineTest, AddDecoratorInvokesHooksAroundStageExecution) {
    PipelineContext context;
    ::testing::StrictMock<MockDatasetLoader> loader;
    EXPECT_CALL(loader, load()).WillOnce(::testing::Return(CalibrationDataset{}));

    CalibrationPipeline pipeline;

    auto stage = std::make_unique<::testing::StrictMock<MockStage>>();
    auto* stage_ptr = stage.get();
    EXPECT_CALL(*stage_ptr, name()).WillRepeatedly(::testing::Return("decorated-stage"));
    EXPECT_CALL(*stage_ptr, run(::testing::Ref(context))).WillOnce([](PipelineContext&) {
        PipelineStageResult result;
        result.name = "decorated-stage";
        result.success = true;
        return result;
    });
    pipeline.add_stage(std::move(stage));

    auto decorator = std::make_shared<::testing::StrictMock<MockDecorator>>();
    auto* decorator_ptr = decorator.get();
    pipeline.add_decorator(decorator);

    {
        ::testing::InSequence seq;
        EXPECT_CALL(*decorator_ptr, before_stage(::testing::_, ::testing::Ref(context)))
            .WillOnce([&](const CalibrationStage& stage, PipelineContext& ctx) {
                EXPECT_EQ(&stage, static_cast<const CalibrationStage*>(stage_ptr));
                EXPECT_EQ(&ctx, &context);
            });
        EXPECT_CALL(*decorator_ptr,
                    after_stage(::testing::_, ::testing::Ref(context), ::testing::_))
            .WillOnce([&](const CalibrationStage& stage, PipelineContext& ctx,
                          const PipelineStageResult& result) {
                EXPECT_EQ(&stage, static_cast<const CalibrationStage*>(stage_ptr));
                EXPECT_EQ(&ctx, &context);
                EXPECT_TRUE(result.success);
            });
    }

    const auto report = pipeline.execute(loader, context);

    EXPECT_TRUE(report.success);
    ASSERT_EQ(report.stages.size(), 1);
    EXPECT_EQ(report.stages.front().name, "decorated-stage");
}

TEST(CalibrationPipelineTest, ExecuteAggregatesStageResultsAndStatus) {
    PipelineContext context;
    ::testing::StrictMock<MockDatasetLoader> loader;
    EXPECT_CALL(loader, load()).WillOnce(::testing::Return(CalibrationDataset{}));

    CalibrationPipeline pipeline;

    auto stage_success = std::make_unique<::testing::StrictMock<MockStage>>();
    EXPECT_CALL(*stage_success, name()).WillRepeatedly(::testing::Return("first"));
    EXPECT_CALL(*stage_success, run(::testing::Ref(context))).WillOnce([](PipelineContext&) {
        PipelineStageResult result;
        result.name = "first";
        result.success = true;
        return result;
    });
    pipeline.add_stage(std::move(stage_success));

    auto stage_failure = std::make_unique<::testing::StrictMock<MockStage>>();
    EXPECT_CALL(*stage_failure, name()).WillRepeatedly(::testing::Return("second"));
    EXPECT_CALL(*stage_failure, run(::testing::Ref(context))).WillOnce([](PipelineContext&) {
        PipelineStageResult result;
        result.success = false;  // leave name empty to exercise fallback
        return result;
    });
    pipeline.add_stage(std::move(stage_failure));

    const auto report = pipeline.execute(loader, context);

    EXPECT_FALSE(report.success);
    ASSERT_EQ(report.stages.size(), 2);
    EXPECT_EQ(report.stages.front().name, "first");
    EXPECT_TRUE(report.stages.front().success);
    EXPECT_EQ(report.stages.back().name, "second");
    EXPECT_FALSE(report.stages.back().success);
}

TEST(LoggingDecoratorTest, BeforeStageWritesMessage) {
    PipelineContext context;
    ::testing::StrictMock<MockStage> stage;
    EXPECT_CALL(stage, name()).WillRepeatedly(::testing::Return("stage-name"));

    std::ostringstream stream;
    LoggingDecorator decorator(stream);

    decorator.before_stage(stage, context);

    EXPECT_EQ(stream.str(), "[pipeline] → Starting stage 'stage-name'\n");
}

TEST(LoggingDecoratorTest, AfterStageWritesSuccessAndFailureMessages) {
    PipelineContext context;
    ::testing::StrictMock<MockStage> stage;
    EXPECT_CALL(stage, name()).WillRepeatedly(::testing::Return("stage"));

    std::ostringstream stream;
    LoggingDecorator decorator(stream);

    PipelineStageResult success_result;
    success_result.success = true;
    decorator.after_stage(stage, context, success_result);

    PipelineStageResult failure_result;
    failure_result.success = false;
    decorator.after_stage(stage, context, failure_result);

    EXPECT_EQ(stream.str(),
              "[pipeline] ← Completed stage 'stage' (success)\n[pipeline] ← Completed stage "
              "'stage' (failed)\n");
}

}  // namespace
}  // namespace calib::pipeline
