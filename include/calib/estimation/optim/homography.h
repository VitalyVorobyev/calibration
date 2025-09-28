/** @brief Ceres-based homography optimization interfaces */

#pragma once

#include <Eigen/Core>

#include "calib/estimation/linear/homography.h"  // PlanarView
#include "calib/estimation/optim/optimize.h"

namespace calib {

struct HomographyOptions final : public OptimOptions {};

struct OptimizeHomographyResult final : OptimResult {
    Eigen::Matrix3d homography;
};

auto optimize_homography(const PlanarView& data, const Eigen::Matrix3d& init_h,
                         const HomographyOptions& options = {}) -> OptimizeHomographyResult;

inline void to_json(nlohmann::json& j, const HomographyOptions& o) {
    to_json(j, static_cast<const OptimOptions&>(o));
}

inline void from_json(const nlohmann::json& j, HomographyOptions& o) {
    from_json(j, static_cast<OptimOptions&>(o));
}

inline void to_json(nlohmann::json& j, const OptimizeHomographyResult& o) {
    j = nlohmann::json::object();
    j["success"] = o.success;
    j["report"] = o.report;
    j["final_cost"] = o.final_cost;
    if (o.covariance.size() > 0) {
        j["covariance"] = o.covariance;
    }
    j["homography"] = o.homography;
}

inline void from_json(const nlohmann::json& j, OptimizeHomographyResult& o) {
    j.at("success").get_to(o.success);
    j.at("report").get_to(o.report);
    j.at("final_cost").get_to(o.final_cost);
    if (j.contains("covariance")) {
        j.at("covariance").get_to(o.covariance);
    }
    j.at("homography").get_to(o.homography);
}

}  // namespace calib
