#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#include "calib/datasets/planar.h"

using namespace calib::planar;

namespace {

auto make_temp_json_path(std::string_view stem) -> std::filesystem::path {
    const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    auto filename = std::string(stem) + "-" + std::to_string(timestamp) + ".json";
    return std::filesystem::temp_directory_path() / filename;
}

}  // namespace

TEST(PlanarDataset, ValidateAndLoadNewFormat) {
    nlohmann::json dataset = {
        {"image_directory", "./images"},
        {"feature_type", "planar"},
        {"algo_version", "1.0"},
        {"params_hash", "deadbeef"},
        {"sensor_id", "cam0"},
        {"tags", nlohmann::json::array({"synthetic"})},
        {"metadata", {{"detector", {{"name", "synthetic"}}}}},
        {"images",
         nlohmann::json::array({{{"file", "img_0001.png"},
                                 {"points", nlohmann::json::array({{{"x", 100.0},
                                                                    {"y", 200.0},
                                                                    {"id", 0},
                                                                    {"local_x", 0.0},
                                                                    {"local_y", 0.0},
                                                                    {"local_z", 0.0}},
                                                                   {{"x", 150.0},
                                                                    {"y", 200.0},
                                                                    {"id", 1},
                                                                    {"local_x", 0.025},
                                                                    {"local_y", 0.0},
                                                                    {"local_z", 0.0}}})}}})}};

    const auto detections = dataset.get<PlanarDetections>();
    EXPECT_EQ(detections.sensor_id, "cam0");
    EXPECT_EQ(detections.images.size(), 1u);
    EXPECT_EQ(detections.images.front().points.size(), 2u);
    EXPECT_TRUE(detections.tags.count("synthetic"));
    EXPECT_EQ(detections.feature_type, "planar");

    const auto tmp_path = make_temp_json_path("calib_dataset");
    {
        std::ofstream out(tmp_path);
        ASSERT_TRUE(out) << "failed to create temporary dataset file";
        out << dataset.dump(2);
    }

    std::ifstream input(tmp_path);
    ASSERT_TRUE(input) << "failed to open dataset";
    nlohmann::json file_json;
    input >> file_json;
    const auto file_detections = file_json.get<PlanarDetections>();
    EXPECT_EQ(file_detections.sensor_id, "cam0");
    EXPECT_EQ(file_detections.images.size(), 1u);
    EXPECT_EQ(file_detections.images.front().points.size(), 2u);

    std::filesystem::remove(tmp_path);
}

TEST(PlanarDataset, MissingImagesFieldThrows) {
    nlohmann::json invalid = {{"image_directory", "./images"},
                              {"sensor_id", "cam0"},
                              {"images", nlohmann::json::array()}};
    EXPECT_NO_THROW({ (void)invalid.get<PlanarDetections>(); });

    nlohmann::json missing_images = {{"image_directory", "./images"}, {"sensor_id", "cam0"}};
    EXPECT_THROW({ (void)missing_images.get<PlanarDetections>(); }, nlohmann::json::exception);
}

TEST(PlanarDataset, SerializationRoundTrip) {
    PlanarDetections detections;
    detections.image_directory = "./images";
    detections.feature_type = "planar";
    detections.algo_version = "1.2.3";
    detections.params_hash = "cafebabe";
    detections.sensor_id = "cam0";
    detections.tags = {"recorded", "synthetic"};
    detections.metadata = nlohmann::json::object();
    detections.metadata["custom"] = 42;

    PlanarImageDetections image;
    image.file = "view0.png";
    image.points.push_back(PlanarTargetPoint{
        .x = 10.0, .y = 20.0, .id = 5, .local_x = 1.0, .local_y = 2.0, .local_z = 0.3});
    image.points.push_back(PlanarTargetPoint{
        .x = 30.0, .y = 40.0, .id = 6, .local_x = 3.0, .local_y = 4.0, .local_z = 0.6});
    detections.images.push_back(image);

    nlohmann::json json = detections;
    const auto restored = json.get<PlanarDetections>();

    EXPECT_EQ(restored.sensor_id, detections.sensor_id);
    EXPECT_EQ(restored.image_directory, detections.image_directory);
    EXPECT_EQ(restored.feature_type, detections.feature_type);
    EXPECT_EQ(restored.algo_version, detections.algo_version);
    EXPECT_EQ(restored.params_hash, detections.params_hash);
    EXPECT_EQ(restored.images.size(), detections.images.size());
    ASSERT_FALSE(restored.images.empty());
    EXPECT_EQ(restored.images.front().file, detections.images.front().file);
    EXPECT_EQ(restored.images.front().points.size(), detections.images.front().points.size());
    const auto& restored_pt = restored.images.front().points.front();
    EXPECT_DOUBLE_EQ(restored_pt.x, 10.0);
    EXPECT_DOUBLE_EQ(restored_pt.y, 20.0);
    EXPECT_EQ(restored_pt.id, 5);
    EXPECT_DOUBLE_EQ(restored_pt.local_x, 1.0);
    EXPECT_DOUBLE_EQ(restored_pt.local_y, 2.0);
    EXPECT_DOUBLE_EQ(restored_pt.local_z, 0.3);

    EXPECT_EQ(restored.tags, detections.tags);
    ASSERT_TRUE(restored.metadata.is_object());
    EXPECT_EQ(restored.metadata.value("custom", 0), 42);
}
