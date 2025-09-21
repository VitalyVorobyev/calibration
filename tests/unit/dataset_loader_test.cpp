#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

#include "calib/utils/planar_intrinsics_utils.h"

using namespace calib::planar;

namespace {

auto make_temp_json_path(std::string_view stem) -> std::filesystem::path {
    const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    auto filename = std::string(stem) + "-" + std::to_string(timestamp) + ".json";
    return std::filesystem::temp_directory_path() / filename;
}

}  // namespace

TEST(PlanarDataset, ValidateAndLoadNewFormat) {
    nlohmann::json dataset;
    dataset["schema_version"] = 1;
    dataset["feature_type"] = "planar_points";
    dataset["metadata"] = {{"image_directory", "./images"}};
    dataset["metadata"]["detector"] = {{"type", "planar"},
                                           {"name", "synthetic"},
                                           {"version", "1.0"},
                                           {"params_hash", "deadbeef"}};

    nlohmann::json capture;
    capture["sensor_id"] = "cam0";
    capture["frame"] = "img_0001.png";
    capture["tags"] = nlohmann::json::array({"synthetic"});

    nlohmann::json observation;
    observation["target_id"] = "board";
    observation["type"] = "planar_points";
    observation["points"] = nlohmann::json::array({
        {{"id", 0}, {"pixel", {100.0, 200.0}}, {"target", {0.0, 0.0, 0.0}}},
        {{"id", 1}, {"pixel", {150.0, 200.0}}, {"target", {0.025, 0.0, 0.0}}}
    });

    capture["observations"] = nlohmann::json::array({observation});
    dataset["captures"] = nlohmann::json::array({capture});

    std::string error;
    EXPECT_TRUE(validate_planar_dataset(dataset, &error)) << error;

    const auto tmp_path = make_temp_json_path("calib_dataset");
    {
        std::ofstream out(tmp_path);
        ASSERT_TRUE(out) << "failed to create temporary dataset file";
        out << dataset.dump(2);
    }

    const auto detections = load_planar_observations(tmp_path, std::string("cam0"));
    EXPECT_EQ(detections.sensor_id, "cam0");
    EXPECT_EQ(detections.images.size(), 1u);
    EXPECT_EQ(detections.images.front().points.size(), 2u);
    EXPECT_TRUE(detections.tags.count("synthetic"));
    EXPECT_EQ(detections.feature_type, "planar");

    std::filesystem::remove(tmp_path);
}

TEST(PlanarDataset, ConvertLegacyFormat) {
    nlohmann::json legacy;
    legacy["image_directory"] = "./legacy";
    legacy["feature_type"] = "planar";
    legacy["algo_version"] = "opencv-4.8.0";
    legacy["params_hash"] = "cafebabe";

    nlohmann::json legacy_image;
    legacy_image["file"] = "img_0001.png";
    legacy_image["count"] = 2;
    legacy_image["points"] = nlohmann::json::array({
        {{"id", 0}, {"x", 10.0}, {"y", 20.0}, {"local_x", 0.0}, {"local_y", 0.0}, {"local_z", 0.0}},
        {{"id", 1}, {"x", 30.0}, {"y", 40.0}, {"local_x", 0.025}, {"local_y", 0.0}, {"local_z", 0.0}}
    });
    legacy["images"] = nlohmann::json::array({legacy_image});

    auto converted = convert_legacy_planar_features(legacy, "cam_legacy");
    std::string error;
    EXPECT_TRUE(validate_planar_dataset(converted, &error)) << error;
    EXPECT_EQ(converted.at("schema_version"), 1);
    EXPECT_EQ(converted.at("feature_type"), "planar_points");

    const auto tmp_path = make_temp_json_path("legacy_dataset");
    {
        std::ofstream out(tmp_path);
        ASSERT_TRUE(out);
        out << converted.dump(2);
    }

    const auto detections = load_planar_observations(tmp_path);
    EXPECT_EQ(detections.sensor_id, "cam_legacy");
    EXPECT_EQ(detections.images.size(), 1u);
    EXPECT_EQ(detections.images.front().points.size(), 2u);
    EXPECT_TRUE(detections.tags.count("recorded"));

    std::filesystem::remove(tmp_path);
}

TEST(PlanarDataset, ValidateRejectsEmptyCaptures) {
    nlohmann::json invalid = {{"schema_version", 1},
                              {"feature_type", "planar_points"},
                              {"captures", nlohmann::json::array()}};
    std::string error;
    EXPECT_FALSE(validate_planar_dataset(invalid, &error));
    EXPECT_FALSE(error.empty());
}

