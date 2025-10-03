#include "calib/pipeline/loaders.h"

// std
#include <fstream>
#include <stdexcept>

// third-party
#include <nlohmann/json.hpp>

namespace calib::pipeline {

namespace {

struct LoadedPlanarSource final {
    PlanarDetections detections;
    nlohmann::json source_summary;
    nlohmann::json raw_payload;
};

auto load_planar_source(const JsonPlanarDatasetLoader::Entry& entry) -> LoadedPlanarSource {
    std::ifstream stream(entry.path);
    if (!stream) {
        throw std::runtime_error("JsonPlanarDatasetLoader: failed to open " + entry.path.string());
    }

    nlohmann::json json_data;
    stream >> json_data;

    auto detections = json_data.get<PlanarDetections>();
    detections.source_file = entry.path;

    if (entry.sensor_id.has_value() && detections.sensor_id != *entry.sensor_id) {
        throw std::runtime_error("Requested sensor_id '" + *entry.sensor_id +
                                 "' not found in dataset.");
    }

    nlohmann::json source_info{{"path", entry.path.string()}, {"sensor_id", detections.sensor_id}};
    if (!detections.metadata.is_null()) {
        source_info["detector"] = detections.metadata.value("detector", nlohmann::json::object());
    }

    return LoadedPlanarSource{
        .detections = std::move(detections),
        .source_summary = std::move(source_info),
        .raw_payload = std::move(json_data),
    };
}

}  // namespace

void JsonPlanarDatasetLoader::add_entry(const std::filesystem::path& path,
                                        std::optional<std::string> sensor_id) {
    entries_.push_back(Entry{path, std::move(sensor_id)});
}

auto JsonPlanarDatasetLoader::load() -> CalibrationDataset {
    if (entries_.empty()) {
        throw std::runtime_error("JsonPlanarDatasetLoader: no dataset entries configured.");
    }

    CalibrationDataset dataset;
    dataset.metadata = nlohmann::json::object();
    dataset.metadata["sources"] = nlohmann::json::array();
    dataset.raw_json = nlohmann::json::object();

    for (const auto& entry : entries_) {
        auto loaded = load_planar_source(entry);
        dataset.metadata["sources"].push_back(std::move(loaded.source_summary));
        dataset.raw_json[entry.path.string()] = std::move(loaded.raw_payload);
        dataset.planar_cameras.push_back(std::move(loaded.detections));
    }

    dataset.schema_version = 1;
    return dataset;
}

}  // namespace calib::pipeline
