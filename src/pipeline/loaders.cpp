#include "calib/pipeline/loaders.h"

// std
#include <stdexcept>

namespace calib::pipeline {

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
    dataset.raw_json = nlohmann::json::object();

    nlohmann::json sources = nlohmann::json::array();
    for (const auto& entry : entries_) {
        auto detections = planar::load_planar_dataset(entry.path, entry.sensor_id);
        nlohmann::json source_info{{"path", entry.path.string()},
                                   {"sensor_id", detections.sensor_id}};
        if (!detections.metadata.is_null()) {
            source_info["detector"] =
                detections.metadata.value("detector", nlohmann::json::object());
        }
        sources.push_back(source_info);
        dataset.planar_cameras.push_back(std::move(detections));
    }

    dataset.metadata["sources"] = std::move(sources);
    dataset.schema_version = 1;
    return dataset;
}

}  // namespace calib::pipeline
