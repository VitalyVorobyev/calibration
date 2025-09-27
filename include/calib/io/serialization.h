#pragma once

#include <Eigen/Geometry>
#include <nlohmann/json.hpp>

#include "calib/io/json.h"
#include <vector>

#include "calib/estimation/bundle.h"
#include "calib/estimation/extrinsics.h"
#include "calib/estimation/handeye.h"
#include "calib/estimation/intrinsics.h"
#include "calib/estimation/planarpose.h"
#include "calib/models/camera_matrix.h"
#include "calib/models/distortion.h"
#include "calib/models/pinhole.h"

// Make Eigen types natively serializable for nlohmann::json
namespace nlohmann {

template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct adl_serializer<Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>> {
    static void to_json(json& j,
                        const Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& m) {
        if (m.rows() == 1 || m.cols() == 1) {
            j = json::array();
            const auto n = static_cast<size_t>(m.size());
            for (size_t i = 0; i < n; ++i) j.push_back(m(Eigen::Index(i)));
        } else {
            j = json::array();
            for (Eigen::Index r = 0; r < m.rows(); ++r) {
                json row = json::array();
                for (Eigen::Index c = 0; c < m.cols(); ++c) row.push_back(m(r, c));
                j.push_back(row);
            }
        }
    }

    static void from_json(const json& j,
                          Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& m) {
        if (!j.is_array()) throw std::runtime_error("Eigen JSON: expected array");
        if (!j.empty() && j.front().is_array()) {
            const Eigen::Index rows = static_cast<Eigen::Index>(j.size());
            const Eigen::Index cols = static_cast<Eigen::Index>(j.front().size());
            m.resize(rows, cols);
            for (Eigen::Index r = 0; r < rows; ++r)
                for (Eigen::Index c = 0; c < cols; ++c) m(r, c) = j[r][c].get<Scalar>();
        } else {
            const Eigen::Index n = static_cast<Eigen::Index>(j.size());
            if constexpr (Cols == 1 || Rows == 1) {
                m.resize((Rows == 1 ? 1 : n), (Cols == 1 ? 1 : n));
            } else {
                m.resize(n, 1);
            }
            for (Eigen::Index i = 0; i < n; ++i) m(Eigen::Index(i)) = j[i].get<Scalar>();
        }
    }
};

template <typename Scalar, int Dim, int Mode, int Options>
struct adl_serializer<Eigen::Transform<Scalar, Dim, Mode, Options>> {
    static void to_json(json& j, const Eigen::Transform<Scalar, Dim, Mode, Options>& T) {
        const auto M = T.matrix();
        j = M;
    }
    static void from_json(const json& j, Eigen::Transform<Scalar, Dim, Mode, Options>& T) {
        Eigen::Matrix<Scalar, Dim + 1, Dim + 1> M = j.get<Eigen::Matrix<Scalar, Dim + 1, Dim + 1>>();
        T.matrix() = M;
    }
};

}  // namespace nlohmann

namespace calib {

// ----- Basic structures -----

inline void to_json(nlohmann::json& j, const DualDistortion& d) {
    j = {{"forward", d.forward}, {"inverse", d.inverse}};
}

inline void from_json(const nlohmann::json& j, DualDistortion& d) {
    if (j.contains("forward")) d.forward = j.at("forward").get<Eigen::VectorXd>();
    if (j.contains("inverse")) d.inverse = j.at("inverse").get<Eigen::VectorXd>();
}

inline void to_json(nlohmann::json& j, const BrownConradyd& d) { j = {{"coeffs", d.coeffs}}; }

inline void from_json(const nlohmann::json& j, BrownConradyd& d) {
    if (j.contains("coeffs")) d.coeffs = j.at("coeffs").get<Eigen::VectorXd>();
}

template <distortion_model DistortionT>
inline void to_json(nlohmann::json& j, const PinholeCamera<DistortionT>& cam) {
    j = {{"kmtx", cam.kmtx}, {"distortion", cam.distortion}};
}

template <distortion_model DistortionT>
inline void from_json(const nlohmann::json& j, PinholeCamera<DistortionT>& cam) {
    j.at("kmtx").get_to(cam.kmtx);
    if (j.contains("distortion")) j.at("distortion").get_to(cam.distortion);
}

inline void to_json(nlohmann::json& j, const PlanarObservation& p) {
    j = {{"object", {p.object_xy.x(), p.object_xy.y()}},
         {"image", {p.image_uv.x(), p.image_uv.y()}}};
}

inline void from_json(const nlohmann::json& j, PlanarObservation& p) {
    auto obj = j.at("object");
    auto img = j.at("image");
    p.object_xy = Eigen::Vector2d(obj[0].get<double>(), obj[1].get<double>());
    p.image_uv = Eigen::Vector2d(img[0].get<double>(), img[1].get<double>());
}

inline void to_json(nlohmann::json& j, const BundleOptions& o) {
    j = {{"optimize_intrinsics", o.optimize_intrinsics},
         {"optimize_skew", o.optimize_skew},
         {"optimize_target_pose", o.optimize_target_pose},
         {"optimize_hand_eye", o.optimize_hand_eye},
         {"verbose", o.verbose}};
}

inline void from_json(const nlohmann::json& j, BundleOptions& o) {
    o.optimize_intrinsics = j.value("optimize_intrinsics", false);
    o.optimize_skew = j.value("optimize_skew", false);
    o.optimize_target_pose = j.value("optimize_target_pose", true);
    o.optimize_hand_eye = j.value("optimize_hand_eye", true);
    o.verbose = j.value("verbose", false);
}

inline void to_json(nlohmann::json& j, const BundleObservation& bo) {
    j = {{"view", bo.view}, {"b_se3_g", bo.b_se3_g}, {"camera_index", bo.camera_index}};
}

inline void from_json(const nlohmann::json& j, BundleObservation& bo) {
    j.at("view").get_to(bo.view);
    bo.b_se3_g = j.at("b_se3_g").get<Eigen::Isometry3d>();
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
    std::vector<PinholeCamera<DualDistortion>> cameras;
    std::vector<MulticamPlanarView> views;
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
    std::vector<PinholeCamera<BrownConradyd>> initial_cameras;
    std::vector<Eigen::Isometry3d> init_g_se3_c;
    Eigen::Isometry3d init_b_se3_t = Eigen::Isometry3d::Identity();
    BundleOptions options;
};

inline void to_json(nlohmann::json& j, const BundleInput& in) {
    j = {{"observations", in.observations},
         {"initial_cameras", in.initial_cameras},
         {"init_g_se3_c", in.init_g_se3_c},
         {"init_b_se3_t", in.init_b_se3_t},
         {"options", in.options}};
}

inline void from_json(const nlohmann::json& j, BundleInput& in) {
    j.at("observations").get_to(in.observations);
    j.at("initial_cameras").get_to(in.initial_cameras);
    in.init_g_se3_c.clear();
    if (j.contains("init_g_se3_c"))
        for (const auto& jt : j.at("init_g_se3_c")) in.init_g_se3_c.push_back(jt.get<Eigen::Isometry3d>());
    if (j.contains("init_b_se3_t")) in.init_b_se3_t = j.at("init_b_se3_t").get<Eigen::Isometry3d>();
    if (j.contains("options")) j.at("options").get_to(in.options);
}

// ----- Result serialization -----

template <camera_model CameraT>
inline void to_json(nlohmann::json& j, const IntrinsicsOptimizationResult<CameraT>& r) {
    j = {{"camera", r.camera},
         {"poses", r.c_se3_t},
         {"covariance", r.covariance},
         {"view_errors", r.view_errors},
         {"final_cost", r.final_cost},
         {"report", r.report}};
}

template <camera_model CameraT>
inline void from_json(const nlohmann::json& j, IntrinsicsOptimizationResult<CameraT>& r) {
    j.at("camera").get_to(r.camera);
    r.c_se3_t = j.value("poses", std::vector<Eigen::Isometry3d>{});
    r.covariance = j.at("covariance").get<Eigen::MatrixXd>();
    r.view_errors = j.value("view_errors", std::vector<double>{});
    r.final_cost = j.value("final_cost", 0.0);
    r.report = j.value("report", std::string{});
}

template <camera_model CameraT>
inline void to_json(nlohmann::json& j, const ExtrinsicOptimizationResult<CameraT>& r) {
    j = {{"cameras", r.cameras},
         {"c_se3_r", r.c_se3_r},
         {"r_se3_t", r.r_se3_t},
         {"covariance", r.covariance},
         {"final_cost", r.final_cost},
         {"report", r.report}};
}

template <camera_model CameraT>
inline void from_json(const nlohmann::json& j, ExtrinsicOptimizationResult<CameraT>& r) {
    r.c_se3_r = j.value("c_se3_r", std::vector<Eigen::Isometry3d>{});
    r.r_se3_t = j.value("r_se3_t", std::vector<Eigen::Isometry3d>{});
    r.cameras = j.value("cameras", std::vector<CameraT>{});
    r.covariance = j.at("covariance").get<Eigen::MatrixXd>();
    r.final_cost = j.value("final_cost", 0.0);
    r.report = j.value("report", std::string{});
}

inline void to_json(nlohmann::json& j, const BundleResult<PinholeCamera<BrownConradyd>>& r) {
    j = {{"cameras", r.cameras},
         {"g_se3_c", r.g_se3_c},
         {"b_se3_t", r.b_se3_t},
         {"final_cost", r.final_cost},
         {"covariance", r.covariance},
         {"report", r.report}};
}

inline void from_json(const nlohmann::json& j, BundleResult<PinholeCamera<BrownConradyd>>& r) {
    r.cameras = j.value("cameras", std::vector<PinholeCamera<BrownConradyd>>{});
    r.g_se3_c = j.value("g_se3_c", std::vector<Eigen::Isometry3d>{});
    r.b_se3_t = j.at("b_se3_t").get<Eigen::Isometry3d>();
    r.final_cost = j.value("final_cost", 0.0);
    r.covariance = j.at("covariance").get<Eigen::MatrixXd>();
    r.report = j.value("report", std::string{});
}

}  // namespace calib
