#pragma once

// std
#include <filesystem>
#include <optional>
#include <vector>

#include "calib/pipeline/pipeline.h"

namespace calib::pipeline {

class JsonPlanarDatasetLoader final : public DatasetLoader {
  public:
    struct Entry final {
        std::filesystem::path path;
        std::optional<std::string> sensor_id;
    };

  private:
    std::vector<Entry> entries_;

  public:
    JsonPlanarDatasetLoader() = default;
    explicit JsonPlanarDatasetLoader(std::vector<Entry> entries) : entries_(std::move(entries)) {}

    void add_entry(const std::filesystem::path& path,
                   std::optional<std::string> sensor_id = std::nullopt);

    [[nodiscard]] auto load() -> CalibrationDataset override;
};

}  // namespace calib::pipeline
