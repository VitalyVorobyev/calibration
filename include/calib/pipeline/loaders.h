#pragma once

// std
#include <filesystem>
#include <optional>
#include <vector>

#include "calib/pipeline/pipeline.h"

namespace calib::pipeline {

/**
 * @brief Loader that reads planar target detections from JSON files.
 *
 * The loader accepts one or more dataset descriptions on disk. Each entry is
 * validated when @ref load() is invoked and the resulting detections are
 * grouped into a single @ref CalibrationDataset.
 */
class JsonPlanarDatasetLoader final : public DatasetLoader {
  public:
    /**
     * @brief Description of a JSON dataset entry on disk.
     */
    struct Entry final {
        std::filesystem::path path;
        std::optional<std::string> sensor_id;
    };

  private:
    std::vector<Entry> entries_;

  public:
    JsonPlanarDatasetLoader() = default;
    explicit JsonPlanarDatasetLoader(std::vector<Entry> entries) : entries_(std::move(entries)) {}

    /**
     * @brief Append a dataset file to the loader queue.
     *
     * @param path      Filesystem location of the JSON payload to consume.
     * @param sensor_id Optional identifier to match against the dataset's
     *                  @c sensor_id. When specified the loader throws if the
     *                  file contains detections from a different sensor.
     */
    void add_entry(const std::filesystem::path& path,
                   std::optional<std::string> sensor_id = std::nullopt);

    /**
     * @brief Load and validate all configured dataset entries.
     *
     * @throws std::runtime_error if no entries were configured or if any entry
     *         cannot be opened / validated.
     */
    [[nodiscard]] auto load() -> CalibrationDataset override;
};

}  // namespace calib::pipeline
