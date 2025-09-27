#pragma once

#include <array>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "calib/config/planar_intrinsics.h"
#include "calib/models/camera_matrix.h"

namespace calib::planar {

struct SessionReport {
    std::string id;
    std::string description;
    std::string timestamp_utc;
};

struct PlanarIntrinsicsOptionsReport {
    std::size_t min_corners_per_view = 0;
    bool refine = false;
    bool optimize_skew = false;
    int num_radial = 0;
    double huber_delta = 0.0;
    int max_iterations = 0;
    double epsilon = 0.0;
    double point_scale = 1.0;
    bool auto_center_points = true;
    std::optional<std::array<double, 2>> point_center;
    std::vector<int> fixed_distortion_indices;
    std::vector<double> fixed_distortion_values;
    std::optional<HomographyRansacConfig> homography_ransac;
};

struct InitialGuessWarningCounts {
    std::size_t invalid_camera_matrix = 0;
    std::size_t homography_decomposition_failures = 0;
};

struct InitialGuessReport {
    CameraMatrix intrinsics;
    std::vector<std::size_t> used_view_indices;
    InitialGuessWarningCounts warning_counts;
};

struct PlanarViewReport {
    std::string source_image;
    std::size_t corner_count = 0;
    double rms_px = 0.0;
    bool used_in_linear_stage = false;
};

struct IntrinsicsResultReport {
    CameraMatrix intrinsics;
    std::string distortion_model;
    std::vector<double> distortion_coefficients;
    double reprojection_rms_px = 0.0;
    std::vector<PlanarViewReport> per_view;
};

struct CameraReport {
    std::string camera_id;
    std::string model;
    std::optional<std::array<int, 2>> image_size;
    InitialGuessReport initial_guess;
    IntrinsicsResultReport result;
};

struct CalibrationReport final {
    std::string type;
    std::string algorithm;
    PlanarIntrinsicsOptionsReport options;
    nlohmann::json detector;
    std::vector<CameraReport> cameras;
};

struct PlanarIntrinsicsReport final {
    SessionReport session;
    std::vector<CalibrationReport> calibrations;
};

void to_json(nlohmann::json& json, const SessionReport& report);
void from_json(const nlohmann::json& json, SessionReport& report);

void to_json(nlohmann::json& json, const PlanarIntrinsicsOptionsReport& report);
void from_json(const nlohmann::json& json, PlanarIntrinsicsOptionsReport& report);

void to_json(nlohmann::json& json, const InitialGuessWarningCounts& report);
void from_json(const nlohmann::json& json, InitialGuessWarningCounts& report);

void to_json(nlohmann::json& json, const InitialGuessReport& report);
void from_json(const nlohmann::json& json, InitialGuessReport& report);

void to_json(nlohmann::json& json, const PlanarViewReport& report);
void from_json(const nlohmann::json& json, PlanarViewReport& report);

void to_json(nlohmann::json& json, const IntrinsicsResultReport& report);
void from_json(const nlohmann::json& json, IntrinsicsResultReport& report);

void to_json(nlohmann::json& json, const CameraReport& report);
void from_json(const nlohmann::json& json, CameraReport& report);

void to_json(nlohmann::json& json, const CalibrationReport& report);
void from_json(const nlohmann::json& json, CalibrationReport& report);

void to_json(nlohmann::json& json, const PlanarIntrinsicsReport& report);
void from_json(const nlohmann::json& json, PlanarIntrinsicsReport& report);

}  // namespace calib::planar
