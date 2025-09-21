#include "calib/datasets/planar.h"

#include <fstream>
#include <sstream>

namespace calib::planar {

namespace {

[[nodiscard]] auto collect_tags(const nlohmann::json& capture) -> std::set<std::string> {
    std::set<std::string> tags;
    if (const auto it = capture.find("tags"); it != capture.end() && it->is_array()) {
        for (const auto& tag : *it) {
            if (tag.is_string()) {
                tags.insert(tag.get<std::string>());
            }
        }
    }
    return tags;
}

[[nodiscard]] auto point_from_json(const nlohmann::json& pt_json) -> PlanarTargetPoint {
    PlanarTargetPoint pt;
    if (const auto it = pt_json.find("id"); it != pt_json.end() && it->is_number_integer()) {
        pt.id = it->get<int>();
    }
    const auto& pixel = pt_json.at("pixel");
    pt.x = pixel[0].get<double>();
    pt.y = pixel[1].get<double>();
    const auto& target = pt_json.at("target");
    pt.local_x = target[0].get<double>();
    pt.local_y = target[1].get<double>();
    pt.local_z = target.size() > 2 ? target[2].get<double>() : 0.0;
    return pt;
}

}  // namespace

auto validate_planar_dataset(const nlohmann::json& dataset, std::string* error_message) -> bool {
    const auto set_error = [&](std::string_view message) {
        if (error_message != nullptr) {
            *error_message = std::string(message);
        }
        return false;
    };

    if (!dataset.is_object()) {
        return set_error("Dataset must be a JSON object.");
    }
    if (!dataset.contains("schema_version")) {
        return set_error("Missing schema_version field.");
    }
    if (!dataset.at("schema_version").is_number_integer()) {
        return set_error("schema_version must be an integer.");
    }
    const int schema_version = dataset.at("schema_version").get<int>();
    if (schema_version != 1) {
        return set_error("Unsupported dataset schema_version. Only version 1 is supported.");
    }
    if (!dataset.contains("captures") || !dataset.at("captures").is_array()) {
        return set_error("captures must be an array.");
    }
    if (dataset.at("captures").empty()) {
        return set_error("captures array cannot be empty.");
    }

    for (const auto& capture : dataset.at("captures")) {
        if (!capture.is_object()) {
            return set_error("Each capture must be an object.");
        }
        const auto sid_it = capture.find("sensor_id");
        if (sid_it == capture.end() || !sid_it->is_string() || sid_it->get<std::string>().empty()) {
            return set_error("capture.sensor_id must be a non-empty string.");
        }
        const auto frame_it = capture.find("frame");
        if (frame_it == capture.end() || !frame_it->is_string() ||
            frame_it->get<std::string>().empty()) {
            return set_error("capture.frame must be a non-empty string.");
        }
        const auto obs_it = capture.find("observations");
        if (obs_it == capture.end() || !obs_it->is_array() || obs_it->empty()) {
            return set_error("capture.observations must be a non-empty array.");
        }
        for (const auto& obs : *obs_it) {
            if (!obs.is_object()) {
                return set_error("observation entries must be objects.");
            }
            const auto type_it = obs.find("type");
            if (type_it == obs.end() || !type_it->is_string()) {
                return set_error("observation.type must be a string.");
            }
            if (obs.find("target_id") == obs.end() || !obs.at("target_id").is_string()) {
                return set_error("observation.target_id must be a string.");
            }
            const auto points_it = obs.find("points");
            if (points_it == obs.end() || !points_it->is_array() || points_it->empty()) {
                return set_error("observation.points must be a non-empty array.");
            }
            for (const auto& pt : *points_it) {
                if (!pt.is_object()) {
                    return set_error("observation.points entries must be objects.");
                }
                const auto pix_it = pt.find("pixel");
                if (pix_it == pt.end() || !pix_it->is_array() || pix_it->size() != 2) {
                    return set_error("point.pixel must be a [u, v] array.");
                }
                if (!(*pix_it)[0].is_number() || !(*pix_it)[1].is_number()) {
                    return set_error("point.pixel values must be numeric.");
                }
                const auto tar_it = pt.find("target");
                if (tar_it == pt.end() || !tar_it->is_array() ||
                    (tar_it->size() != 2 && tar_it->size() != 3)) {
                    return set_error("point.target must be [X, Y] or [X, Y, Z].");
                }
                if (!(*tar_it)[0].is_number() || !(*tar_it)[1].is_number()) {
                    return set_error("point.target entries must be numeric.");
                }
                if (tar_it->size() == 3 && !(*tar_it)[2].is_number()) {
                    return set_error("point.target[2] must be numeric.");
                }
            }
        }
    }

    return true;
}

auto convert_legacy_planar_features(const nlohmann::json& legacy,
                                    const std::string& sensor_id_hint) -> nlohmann::json {
    nlohmann::json dataset;
    dataset["schema_version"] = 1;
    dataset["feature_type"] = "planar_points";

    nlohmann::json metadata;
    metadata["image_directory"] = legacy.value("image_directory", "");
    nlohmann::json detector;
    detector["type"] = legacy.value("feature_type", "planar");
    detector["version"] = legacy.value("algo_version", "");
    detector["params_hash"] = legacy.value("params_hash", "");
    metadata["detector"] = detector;
    dataset["metadata"] = metadata;

    nlohmann::json captures = nlohmann::json::array();
    const auto& images = legacy.value("images", nlohmann::json::array());
    for (const auto& img : images) {
        if (!img.contains("points") || !img.at("points").is_array()) {
            continue;
        }
        const auto& points = img.at("points");
        if (points.empty()) {
            continue;
        }
        nlohmann::json capture;
        capture["sensor_id"] = sensor_id_hint;
        capture["frame"] = img.value("file", "");
        capture["tags"] = nlohmann::json::array({"recorded"});

        nlohmann::json observation;
        observation["target_id"] = "legacy_planar_target";
        observation["type"] = "planar_points";

        nlohmann::json planar_points = nlohmann::json::array();
        for (const auto& pt : points) {
            nlohmann::json point;
            if (pt.contains("id")) {
                point["id"] = pt.value("id", -1);
            }
            point["pixel"] = {pt.value("x", 0.0), pt.value("y", 0.0)};
            point["target"] = {pt.value("local_x", 0.0), pt.value("local_y", 0.0),
                               pt.value("local_z", 0.0)};
            planar_points.push_back(point);
        }

        if (planar_points.empty()) {
            continue;
        }

        observation["points"] = std::move(planar_points);
        capture["observations"] = nlohmann::json::array({observation});
        captures.push_back(std::move(capture));
    }

    dataset["captures"] = std::move(captures);
    return dataset;
}

auto load_planar_dataset(const std::filesystem::path& path,
                         std::optional<std::string> sensor_filter) -> PlanarDetections {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Failed to open features JSON: " + path.string());
    }

    nlohmann::json json_data;
    stream >> json_data;

    nlohmann::json dataset_json = json_data;
    if (!dataset_json.contains("schema_version")) {
        dataset_json = convert_legacy_planar_features(json_data, sensor_filter.value_or("cam0"));
    }

    std::string validation_error;
    if (!validate_planar_dataset(dataset_json, &validation_error)) {
        throw std::runtime_error("Invalid calibration dataset: " + validation_error);
    }

    PlanarDetections detections;
    detections.source_file = path;
    detections.feature_type = dataset_json.value("feature_type", std::string("planar_points"));
    detections.algo_version.clear();
    detections.params_hash.clear();
    detections.image_directory.clear();
    detections.images.clear();
    detections.tags.clear();
    detections.metadata = nlohmann::json::object();

    if (const auto meta_it = dataset_json.find("metadata"); meta_it != dataset_json.end()) {
        detections.metadata = *meta_it;
        detections.image_directory = meta_it->value("image_directory", "");
        if (const auto det_it = meta_it->find("detector"); det_it != meta_it->end()) {
            detections.feature_type = det_it->value("type", detections.feature_type);
            detections.algo_version = det_it->value("version", detections.algo_version);
            detections.params_hash = det_it->value("params_hash", detections.params_hash);
        }
    }

    std::set<std::string> sensors_present;
    for (const auto& capture : dataset_json.at("captures")) {
        sensors_present.insert(capture.at("sensor_id").get<std::string>());
    }

    std::string selected_sensor;
    if (sensor_filter.has_value()) {
        selected_sensor = *sensor_filter;
        if (!sensors_present.count(selected_sensor)) {
            throw std::runtime_error("Requested sensor_id '" + selected_sensor +
                                     "' not found in dataset.");
        }
    } else if (sensors_present.size() == 1) {
        selected_sensor = *sensors_present.begin();
    } else {
        std::ostringstream oss;
        oss << "Dataset contains multiple sensors (";
        bool first = true;
        for (const auto& sid : sensors_present) {
            if (!first) {
                oss << ", ";
            }
            first = false;
            oss << sid;
        }
        oss << ") but no sensor_id filter was provided.";
        throw std::runtime_error(oss.str());
    }

    detections.sensor_id = selected_sensor;

    const auto& captures = dataset_json.at("captures");
    for (const auto& capture : captures) {
        const auto sensor_id = capture.at("sensor_id").get<std::string>();
        if (sensor_id != selected_sensor) {
            continue;
        }
        const auto capture_tags = collect_tags(capture);
        detections.tags.insert(capture_tags.begin(), capture_tags.end());

        PlanarImageDetections img;
        img.file = capture.value("frame", "");

        const auto& observations = capture.at("observations");
        for (const auto& obs : observations) {
            if (obs.value("type", std::string{}) != "planar_points") {
                continue;
            }
            const auto& points = obs.at("points");
            for (const auto& pt_json : points) {
                img.points.push_back(point_from_json(pt_json));
            }
        }

        img.count = static_cast<int>(img.points.size());
        if (img.count > 0) {
            detections.images.push_back(std::move(img));
        }
    }

    if (detections.images.empty()) {
        throw std::runtime_error("No planar observations found for sensor '" + selected_sensor +
                                 "'.");
    }

    return detections;
}

}  // namespace calib::planar
