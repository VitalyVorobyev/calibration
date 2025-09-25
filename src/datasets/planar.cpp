#include "calib/datasets/planar.h"

#include <nlohmann/json.hpp>

namespace calib::planar {

void to_json(nlohmann::json& json, const PlanarTargetPoint& point) {
    json = nlohmann::json{{"x", point.x},
                          {"y", point.y},
                          {"id", point.id},
                          {"local_x", point.local_x},
                          {"local_y", point.local_y},
                          {"local_z", point.local_z}};
}

void from_json(const nlohmann::json& json, PlanarTargetPoint& point) {
    json.at("x").get_to(point.x);
    json.at("y").get_to(point.y);
    point.id = json.value("id", -1);
    json.at("local_x").get_to(point.local_x);
    json.at("local_y").get_to(point.local_y);
    if (json.contains("local_z")) {
        json.at("local_z").get_to(point.local_z);
    } else {
        point.local_z = 0.0;
    }
}

void to_json(nlohmann::json& json, const PlanarImageDetections& detections) {
    json = nlohmann::json{{"file", detections.file}, {"points", detections.points}};
}

void from_json(const nlohmann::json& json, PlanarImageDetections& detections) {
    json.at("file").get_to(detections.file);
    json.at("points").get_to(detections.points);
}

void to_json(nlohmann::json& json, const PlanarDetections& detections) {
    json = nlohmann::json{{"image_directory", detections.image_directory},
                          {"feature_type", detections.feature_type},
                          {"algo_version", detections.algo_version},
                          {"params_hash", detections.params_hash},
                          {"sensor_id", detections.sensor_id},
                          {"tags", detections.tags},
                          {"metadata", detections.metadata},
                          {"images", detections.images}};
}

void from_json(const nlohmann::json& json, PlanarDetections& detections) {
    if (json.contains("image_directory")) {
        json.at("image_directory").get_to(detections.image_directory);
    } else {
        detections.image_directory.clear();
    }
    if (json.contains("feature_type")) {
        json.at("feature_type").get_to(detections.feature_type);
    } else {
        detections.feature_type.clear();
    }
    if (json.contains("algo_version")) {
        json.at("algo_version").get_to(detections.algo_version);
    } else {
        detections.algo_version.clear();
    }
    if (json.contains("params_hash")) {
        json.at("params_hash").get_to(detections.params_hash);
    } else {
        detections.params_hash.clear();
    }
    json.at("sensor_id").get_to(detections.sensor_id);
    if (json.contains("tags")) {
        json.at("tags").get_to(detections.tags);
    } else {
        detections.tags.clear();
    }
    if (json.contains("metadata")) {
        detections.metadata = json.at("metadata");
    } else {
        detections.metadata = nlohmann::json::object();
    }
    json.at("images").get_to(detections.images);
}

}  // namespace calib::planar
