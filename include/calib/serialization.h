#pragma once

#include <nlohmann/json.hpp>
#include <Eigen/Geometry>
#include <vector>

#include "calib/cameramatrix.h"
#include "calib/distortion.h"
#include "calib/camera.h"
#include "calib/planarpose.h"
#include "calib/intrinsics.h"
#include "calib/extrinsics.h"
#include "calib/handeye.h"
#include "calib/bundle.h"

namespace calib {

// ----- Helpers for Eigen types -----

inline nlohmann::json eigen_matrix_to_json(const Eigen::MatrixXd& m) {
    nlohmann::json rows = nlohmann::json::array();
    for (int r = 0; r < m.rows(); ++r) {
        nlohmann::json row = nlohmann::json::array();
        for (int c = 0; c < m.cols(); ++c) row.push_back(m(r, c));
        rows.push_back(row);
    }
    return rows;
}

inline Eigen::MatrixXd json_to_eigen_matrix(const nlohmann::json& j) {
    const int rows = static_cast<int>(j.size());
    const int cols = rows ? static_cast<int>(j[0].size()) : 0;
    Eigen::MatrixXd m(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m(r, c) = j[r][c].get<double>();
    return m;
}

inline nlohmann::json eigen_vector_to_json(const Eigen::VectorXd& v) {
    nlohmann::json arr = nlohmann::json::array();
    for (int i = 0; i < v.size(); ++i) arr.push_back(v[i]);
    return arr;
}

inline Eigen::VectorXd json_to_eigen_vector(const nlohmann::json& j) {
    Eigen::VectorXd v(static_cast<int>(j.size()));
    for (int i = 0; i < v.size(); ++i) v[i] = j[i].get<double>();
    return v;
}

inline nlohmann::json affine_to_json(const Eigen::Affine3d& T) {
    nlohmann::json j = nlohmann::json::array();
    for (int r = 0; r < 4; ++r) {
        nlohmann::json row = nlohmann::json::array();
        for (int c = 0; c < 4; ++c) row.push_back(T.matrix()(r, c));
        j.push_back(row);
    }
    return j;
}

inline Eigen::Affine3d json_to_affine(const nlohmann::json& j) {
    Eigen::Matrix4d m;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            m(r, c) = j[r][c].get<double>();
    Eigen::Affine3d T;
    T.matrix() = m;
    return T;
}

// ----- Basic structures -----

template <typename T>
void to_json(nlohmann::json& j, const Observation<T>& o) {
    j = {{"x", o.x}, {"y", o.y}, {"u", o.u}, {"v", o.v}};
}

template <typename T>
void from_json(const nlohmann::json& j, Observation<T>& o) {
    j.at("x").get_to(o.x);
    j.at("y").get_to(o.y);
    j.at("u").get_to(o.u);
    j.at("v").get_to(o.v);
}

inline void to_json(nlohmann::json& j, const CameraMatrix& c) {
    j = {{"fx", c.fx}, {"fy", c.fy}, {"cx", c.cx}, {"cy", c.cy}};
}

inline void from_json(const nlohmann::json& j, CameraMatrix& c) {
    j.at("fx").get_to(c.fx);
    j.at("fy").get_to(c.fy);
    j.at("cx").get_to(c.cx);
    j.at("cy").get_to(c.cy);
}

inline void to_json(nlohmann::json& j, const DualDistortion& d) {
    j = {
        {"forward", eigen_vector_to_json(d.forward)},
        {"inverse", eigen_vector_to_json(d.inverse)}
    };
}

inline void from_json(const nlohmann::json& j, DualDistortion& d) {
    if (j.contains("forward")) d.forward = json_to_eigen_vector(j.at("forward"));
    if (j.contains("inverse")) d.inverse = json_to_eigen_vector(j.at("inverse"));
}

inline void to_json(nlohmann::json& j, const BrownConradyd& d) {
    j = {
        {"coeffs", eigen_vector_to_json(d.coeffs)}
    };
}

inline void from_json(const nlohmann::json& j, BrownConradyd& d) {
    if (j.contains("coeffs")) d.coeffs = json_to_eigen_vector(j.at("coeffs"));
}

template<distortion_model DistortionT>
inline void to_json(nlohmann::json& j, const Camera<DistortionT>& cam) {
    j = {{"K", cam.K}, {"distortion", cam.distortion}};
}

template<distortion_model DistortionT>
inline void from_json(const nlohmann::json& j, Camera<DistortionT>& cam) {
    j.at("K").get_to(cam.K);
    if (j.contains("distortion")) j.at("distortion").get_to(cam.distortion);
}

inline void to_json(nlohmann::json& j, const PlanarObservation& p) {
    j = {
        {"object", {p.object_xy.x(), p.object_xy.y()}},
        {"image", {p.image_uv.x(), p.image_uv.y()}}
    };
}

inline void from_json(const nlohmann::json& j, PlanarObservation& p) {
    auto obj = j.at("object");
    auto img = j.at("image");
    p.object_xy = Eigen::Vector2d(obj[0].get<double>(), obj[1].get<double>());
    p.image_uv = Eigen::Vector2d(img[0].get<double>(), img[1].get<double>());
}

inline void to_json(nlohmann::json& j, const BundleOptions& o) {
    j = {
        {"optimize_intrinsics", o.optimize_intrinsics},
        {"optimize_target_pose", o.optimize_target_pose},
        {"optimize_hand_eye", o.optimize_hand_eye},
        {"verbose", o.verbose}
    };
}

inline void from_json(const nlohmann::json& j, BundleOptions& o) {
    o.optimize_intrinsics = j.value("optimize_intrinsics", false);
    o.optimize_target_pose = j.value("optimize_target_pose", true);
    o.optimize_hand_eye = j.value("optimize_hand_eye", true);
    o.verbose = j.value("verbose", false);
}

inline void to_json(nlohmann::json& j, const BundleObservation& bo) {
    j = {
        {"view", bo.view},
        {"b_T_g", affine_to_json(bo.b_T_g)},
        {"camera_index", bo.camera_index}
    };
}

inline void from_json(const nlohmann::json& j, BundleObservation& bo) {
    j.at("view").get_to(bo.view);
    bo.b_T_g = json_to_affine(j.at("b_T_g"));
    bo.camera_index = j.value("camera_index", 0);
}

// ----- Input containers -----

struct IntrinsicsInput final {
    std::vector<Observation<double>> observations;
    int num_radial = 2;
};

inline void to_json(nlohmann::json& j, const IntrinsicsInput& in) {
    j = {{"observations", in.observations}, {"num_radial", in.num_radial}};
}

inline void from_json(const nlohmann::json& j, IntrinsicsInput& in) {
    j.at("observations").get_to(in.observations);
    in.num_radial = j.value("num_radial", 2);
}

struct ExtrinsicsInput final {
    std::vector<Camera<DualDistortion>> cameras;
    std::vector<ExtrinsicPlanarView> views;
};

inline void to_json(nlohmann::json& j, const ExtrinsicsInput& in) {
    j = {{"cameras", in.cameras}, {"views", in.views}};
}

inline void from_json(const nlohmann::json& j, ExtrinsicsInput& in) {
    j.at("cameras").get_to(in.cameras);
    j.at("views").get_to(in.views);
}

struct BundleInput final {
    std::vector<BundleObservation> observations;
    std::vector<Camera<BrownConradyd>> initial_cameras;
    std::vector<Eigen::Affine3d> init_g_T_c;
    Eigen::Affine3d init_b_T_t = Eigen::Affine3d::Identity();
    BundleOptions options;
};

inline void to_json(nlohmann::json& j, const BundleInput& in) {
    nlohmann::json gtc = nlohmann::json::array();
    for (const auto& T : in.init_g_T_c) gtc.push_back(affine_to_json(T));
    j = {
        {"observations", in.observations},
        {"initial_cameras", in.initial_cameras},
        {"init_g_T_c", gtc},
        {"init_b_T_t", affine_to_json(in.init_b_T_t)},
        {"options", in.options}
    };
}

inline void from_json(const nlohmann::json& j, BundleInput& in) {
    j.at("observations").get_to(in.observations);
    j.at("initial_cameras").get_to(in.initial_cameras);
    in.init_g_T_c.clear();
    if (j.contains("init_g_T_c"))
        for (const auto& jt : j.at("init_g_T_c")) in.init_g_T_c.push_back(json_to_affine(jt));
    if (j.contains("init_b_T_t")) in.init_b_T_t = json_to_affine(j.at("init_b_T_t"));
    if (j.contains("options")) j.at("options").get_to(in.options);
}

// ----- Result serialization -----

inline void to_json(nlohmann::json& j, const IntrinsicOptimizationResult& r) {
    j = {
        {"camera", r.camera},
        {"covariance", eigen_matrix_to_json(r.covariance)},
        {"reprojection_error", r.reprojection_error},
        {"summary", r.summary}
    };
}

inline void from_json(const nlohmann::json& j, IntrinsicOptimizationResult& r) {
    j.at("camera").get_to(r.camera);
    r.covariance = json_to_eigen_matrix(j.at("covariance"));
    r.reprojection_error = j.value("reprojection_error", 0.0);
    r.summary = j.value("summary", std::string{});
}

inline void to_json(nlohmann::json& j, const ExtrinsicOptimizationResult& r) {
    nlohmann::json cps = nlohmann::json::array();
    for (const auto& T : r.camera_poses) cps.push_back(affine_to_json(T));
    nlohmann::json tps = nlohmann::json::array();
    for (const auto& T : r.target_poses) tps.push_back(affine_to_json(T));
    nlohmann::json ccov = nlohmann::json::array();
    for (const auto& C : r.camera_covariances) ccov.push_back(eigen_matrix_to_json(C));
    nlohmann::json tcov = nlohmann::json::array();
    for (const auto& C : r.target_covariances) tcov.push_back(eigen_matrix_to_json(C));
    j = {
        {"camera_poses", cps},
        {"target_poses", tps},
        {"camera_covariances", ccov},
        {"target_covariances", tcov},
        {"reprojection_error", r.reprojection_error},
        {"summary", r.summary}
    };
}

inline void from_json(const nlohmann::json& j, ExtrinsicOptimizationResult& r) {
    r.camera_poses.clear();
    for (const auto& jt : j.at("camera_poses")) r.camera_poses.push_back(json_to_affine(jt));
    r.target_poses.clear();
    for (const auto& jt : j.at("target_poses")) r.target_poses.push_back(json_to_affine(jt));
    r.camera_covariances.clear();
    for (const auto& jc : j.at("camera_covariances")) r.camera_covariances.push_back(json_to_eigen_matrix(jc));
    r.target_covariances.clear();
    for (const auto& jc : j.at("target_covariances")) r.target_covariances.push_back(json_to_eigen_matrix(jc));
    r.reprojection_error = j.value("reprojection_error", 0.0);
    r.summary = j.value("summary", std::string{});
}

inline void to_json(nlohmann::json& j, const BundleResult<Camera<BrownConradyd>>& r) {
    nlohmann::json cams = nlohmann::json::array();
    for (const auto& cam : r.cameras) cams.push_back(cam);
    nlohmann::json gtc = nlohmann::json::array();
    for (const auto& T : r.g_T_c) gtc.push_back(affine_to_json(T));
    j = {
        {"cameras", cams},
        {"g_T_c", gtc},
        {"b_T_t", affine_to_json(r.b_T_t)},
        {"reprojection_error", r.reprojection_error},
        {"covariance", eigen_matrix_to_json(r.covariance)},
        {"summary", r.report}
    };
}

inline void from_json(const nlohmann::json& j, BundleResult<Camera<BrownConradyd>>& r) {
    r.cameras.clear();
    for (const auto& jc : j.at("cameras")) r.cameras.push_back(jc.get<Camera<BrownConradyd>>());
    r.g_T_c.clear();
    for (const auto& jt : j.at("g_T_c")) r.g_T_c.push_back(json_to_affine(jt));
    r.b_T_t = json_to_affine(j.at("b_T_t"));
    r.reprojection_error = j.value("reprojection_error", 0.0);
    r.covariance = json_to_eigen_matrix(j.at("covariance"));
    r.report = j.value("summary", std::string{});
}

} // namespace calib
