#include "calib/reports/planar_intrinsics.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <vector>
namespace calib::planar {

namespace {

[[nodiscard]] auto iso_timestamp_utc() -> std::string {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%FT%TZ");
    return oss.str();
}

}  // namespace

[[nodiscard]] auto compute_global_rms(const CalibrationOutputs& out) -> double {
    const auto& refine = out.refine_result;
    if (refine.view_errors.empty()) {
        return 0.0;
    }
    double sum_sq = 0.0;
    std::size_t total_measurements = 0;
    for (std::size_t i = 0; i < refine.view_errors.size(); ++i) {
        const double view_rms = refine.view_errors[i];
        const std::size_t points =
            i < out.active_views.size() ? out.active_views[i].corner_count : 0;
        const std::size_t measurements = points * 2;
        sum_sq += view_rms * view_rms * static_cast<double>(measurements);
        total_measurements += measurements;
    }
    if (total_measurements == 0) {
        return 0.0;
    }
    return std::sqrt(sum_sq / static_cast<double>(total_measurements));
}

auto build_planar_intrinsics_report(const PlanarCalibrationConfig& cfg, const CameraConfig& cam_cfg,
                                    const PlanarDetections& detections,
                                    const CalibrationOutputs& outputs,
                                    const std::filesystem::path& /*features_path*/)
    -> CalibrationReport {
    CameraReport camera;
    camera.camera_id = cam_cfg.camera_id;
    camera.model = cam_cfg.model;
    camera.image_size = cam_cfg.image_size;

    InitialGuessReport initial_guess;
    initial_guess.intrinsics = outputs.linear_kmtx;
    initial_guess.used_view_indices = outputs.linear_view_indices;
    initial_guess.warning_counts =
        InitialGuessWarningCounts{outputs.invalid_k_warnings, outputs.pose_warnings};
    camera.initial_guess = std::move(initial_guess);

    IntrinsicsResultReport result;
    result.intrinsics = outputs.refine_result.camera.kmtx;
    result.distortion_model = cam_cfg.model;
    result.distortion_coefficients =
        std::vector<double>(outputs.refine_result.camera.distortion.coeffs.data(),
                            outputs.refine_result.camera.distortion.coeffs.data() +
                                outputs.refine_result.camera.distortion.coeffs.size());
    result.reprojection_rms_px = compute_global_rms(outputs);

    for (std::size_t i = 0; i < outputs.active_views.size(); ++i) {
        const auto& view = outputs.active_views[i];
        const double view_rms = i < outputs.refine_result.view_errors.size()
                                    ? outputs.refine_result.view_errors[i]
                                    : 0.0;
        const bool used_in_linear =
            std::find(outputs.linear_view_indices.begin(), outputs.linear_view_indices.end(), i) !=
            outputs.linear_view_indices.end();
        result.per_view.push_back(
            PlanarViewReport{view.source_image, view.corner_count, view_rms, used_in_linear});
    }

    camera.result = std::move(result);

    CalibrationReport calibration;
    calibration.type = "intrinsics";
    calibration.algorithm = cfg.algorithm;
    calibration.options = cfg.options;
    if (!detections.metadata.is_null() && detections.metadata.contains("detector")) {
        calibration.detector = detections.metadata.at("detector");
    } else {
        calibration.detector = nlohmann::json::object();
    }
    calibration.cameras.push_back(std::move(camera));

    return calibration;
}

}  // namespace calib::planar
