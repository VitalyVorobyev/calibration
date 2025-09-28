#include "calib/pipeline/handeye.h"

namespace calib::pipeline {

void to_json(nlohmann::json& j, const HandEyeObservationConfig& cfg) {
    j = nlohmann::json::object();
    if (!cfg.view_id.empty()) {
        j["id"] = cfg.view_id;
    }
    j["base_se3_gripper"] = cfg.base_se3_gripper;
    j["images"] = cfg.images;
}

void from_json(const nlohmann::json& j, HandEyeObservationConfig& cfg) {
    cfg.view_id = j.value("id", std::string{});
    if (j.contains("base_se3_gripper")) {
        cfg.base_se3_gripper = j.at("base_se3_gripper").get<Eigen::Isometry3d>();
    } else if (j.contains("b_T_g")) {
        cfg.base_se3_gripper = j.at("b_T_g").get<Eigen::Isometry3d>();
    } else {
        throw std::runtime_error("HandEyeObservationConfig missing 'base_se3_gripper'");
    }
    if (j.contains("images")) {
        cfg.images = j.at("images").get<std::unordered_map<std::string, std::string>>();
    } else {
        cfg.images.clear();
    }
}

void to_json(nlohmann::json& j, const HandEyeRigConfig& cfg) {
    j = nlohmann::json::object();
    j["rig_id"] = cfg.rig_id;
    j["sensors"] = cfg.sensors;
    j["observations"] = cfg.observations;
    nlohmann::json opts;
    to_json(opts, cfg.options);
    j["options"] = std::move(opts);
    j["min_angle_deg"] = cfg.min_angle_deg;
}

void from_json(const nlohmann::json& j, HandEyeRigConfig& cfg) {
    cfg.rig_id = j.value("rig_id", std::string{});
    j.at("sensors").get_to(cfg.sensors);
    cfg.observations.clear();
    if (j.contains("observations")) {
        cfg.observations = j.at("observations").get<std::vector<HandEyeObservationConfig>>();
    }
    if (j.contains("options")) {
        j.at("options").get_to(cfg.options);
    } else {
        cfg.options = HandeyeOptions{};
    }
    cfg.min_angle_deg = j.value("min_angle_deg", cfg.min_angle_deg);
    if (cfg.rig_id.empty() && !cfg.sensors.empty()) {
        cfg.rig_id = cfg.sensors.front();
    }
}

void to_json(nlohmann::json& j, const HandEyePipelineConfig& cfg) {
    j = nlohmann::json{{"rigs", cfg.rigs}};
}

void from_json(const nlohmann::json& j, HandEyePipelineConfig& cfg) {
    cfg.rigs.clear();
    if (j.contains("rigs")) {
        cfg.rigs = j.at("rigs").get<std::vector<HandEyeRigConfig>>();
    }
}

void to_json(nlohmann::json& j, const BundleRigConfig& cfg) {
    j = nlohmann::json::object();
    j["rig_id"] = cfg.rig_id;
    j["sensors"] = cfg.sensors;
    j["observations"] = cfg.observations;
    nlohmann::json opts;
    to_json(opts, cfg.options);
    j["options"] = std::move(opts);
    j["min_angle_deg"] = cfg.min_angle_deg;
    if (cfg.initial_target.has_value()) {
        j["initial_target"] = cfg.initial_target.value();
    }
}

void from_json(const nlohmann::json& j, BundleRigConfig& cfg) {
    cfg.rig_id = j.value("rig_id", std::string{});
    j.at("sensors").get_to(cfg.sensors);
    cfg.observations.clear();
    if (j.contains("observations")) {
        cfg.observations = j.at("observations").get<std::vector<HandEyeObservationConfig>>();
    }
    if (j.contains("options")) {
        j.at("options").get_to(cfg.options);
    } else {
        cfg.options = BundleOptions{};
    }
    cfg.min_angle_deg = j.value("min_angle_deg", cfg.min_angle_deg);
    if (j.contains("initial_target")) {
        cfg.initial_target = j.at("initial_target").get<Eigen::Isometry3d>();
    } else {
        cfg.initial_target.reset();
    }
    if (cfg.rig_id.empty() && !cfg.sensors.empty()) {
        cfg.rig_id = cfg.sensors.front();
    }
}

void to_json(nlohmann::json& j, const BundlePipelineConfig& cfg) {
    j = nlohmann::json{{"rigs", cfg.rigs}};
}

void from_json(const nlohmann::json& j, BundlePipelineConfig& cfg) {
    cfg.rigs.clear();
    if (j.contains("rigs")) {
        cfg.rigs = j.at("rigs").get<std::vector<BundleRigConfig>>();
    }
}

}  // namespace calib::pipeline
