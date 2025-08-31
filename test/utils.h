/** @brief Utility functions for camera calibration tests */

#pragma once

// std
#include <random>
#include <vector>
#include <cmath>
#include <numbers>

// eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "calib/camera.h"
#include "calib/bundle.h"
#include "calib/scheimpflug.h"

using calib::Camera;
using calib::BrownConradyd;
using calib::PlanarView;
using calib::BundleObservation;
using calib::ScheimpflugCamera;

static inline double deg2rad(double d) { return d * std::numbers::pi / 180.0; }
static inline double rad2deg(double r) { return r * 180.0 / std::numbers::pi; }

inline double rotation_angle(const Eigen::Matrix3d& R) {
    double c = (R.trace() - 1.0) * 0.5;
    c = std::max(-1.0, std::min(1.0, c));
    return std::acos(c); // [0,pi]
}

inline PlanarView make_view(const std::vector<Eigen::Vector2d>& obj,
                            const std::vector<Eigen::Vector2d>& img) {
    PlanarView view(obj.size());
    for (size_t i = 0; i < obj.size(); ++i) {
        view[i].object_xy = obj[i];
        view[i].image_uv = img[i];
    }
    return view;
}

inline Eigen::Affine3d compute_camera_T_target(
    const Eigen::Affine3d& b_T_t,
    const Eigen::Affine3d& g_T_c,
    const Eigen::Affine3d& b_T_g) {
    Eigen::Affine3d c_T_t = g_T_c.inverse() * b_T_g.inverse() * b_T_t;
    return c_T_t;
}

inline Eigen::Matrix3d axis_angle_to_R(const Eigen::Vector3d& axis, double angle){
    if (angle < 1e-16) return Eigen::Matrix3d::Identity();
    return Eigen::AngleAxisd(angle, axis.normalized()).toRotationMatrix();
}

inline Eigen::Affine3d make_pose(const Eigen::Vector3d& t, const Eigen::Vector3d& axis, double angle) {
    Eigen::Affine3d T = Eigen::Affine3d::Identity();
    T.linear() = axis_angle_to_R(axis, angle);
    T.translation() = t;
    return T;
}

inline std::vector<Eigen::Affine3d> make_circle_poses(int n, double radius, double z0,
                                                      double z_step, double rot_step,
                                                      double axis_z = 1.0) {
    std::vector<Eigen::Affine3d> poses;
    poses.reserve(n);
    for (int i = 0; i < n; ++i) {
        double angle = i * 2.0 * std::numbers::pi / n;
        Eigen::Affine3d T = Eigen::Affine3d::Identity();
        T.translation() = Eigen::Vector3d(radius * std::cos(angle),
                                          radius * std::sin(angle),
                                          z0 + z_step * i);
        Eigen::Vector3d axis(std::cos(angle), std::sin(angle), axis_z);
        T.linear() = Eigen::AngleAxisd(rot_step * i, axis.normalized()).toRotationMatrix();
        poses.push_back(T);
    }
    return poses;
}

template <class DistortionT>
inline std::vector<BundleObservation> make_scheimpflug_observations(
    const std::vector<ScheimpflugCamera<DistortionT>>& scs,
    const std::vector<Eigen::Affine3d>& g_T_cs,
    const Eigen::Affine3d& b_T_t,
    const std::vector<Eigen::Vector2d>& obj,
    const std::vector<Eigen::Affine3d>& b_T_gs) {
    std::vector<BundleObservation> obs;
    obs.reserve(b_T_gs.size() * scs.size());
    for (const auto& btg : b_T_gs) {
        for (size_t cam_idx = 0; cam_idx < scs.size(); ++cam_idx) {
            Eigen::Affine3d c_T_t = compute_camera_T_target(b_T_t, g_T_cs[cam_idx], btg);
            std::vector<Eigen::Vector2d> img;
            img.reserve(obj.size());
            for (const auto& xy : obj) {
                Eigen::Vector3d P(xy.x(), xy.y(), 0);
                P = c_T_t * P;
                img.push_back(scs[cam_idx].project(P));
            }
            obs.push_back({make_view(obj, img), btg, static_cast<int>(cam_idx)});
        }
    }
    return obs;
}

template <class DistortionT>
inline std::vector<BundleObservation> make_bundle_observations(
    const std::vector<Camera<DistortionT>>& cams,
    const std::vector<Eigen::Affine3d>& g_T_cs,
    const Eigen::Affine3d& b_T_t,
    const std::vector<Eigen::Vector2d>& obj,
    const std::vector<Eigen::Affine3d>& b_T_gs) {
    std::vector<BundleObservation> obs;
    obs.reserve(b_T_gs.size() * cams.size());
    for (const auto& btg : b_T_gs) {
        for (size_t cam_idx = 0; cam_idx < cams.size(); ++cam_idx) {
            Eigen::Affine3d c_T_t = compute_camera_T_target(b_T_t, g_T_cs[cam_idx], btg);
            std::vector<Eigen::Vector2d> img;
            img.reserve(obj.size());
            for (const auto& xy : obj) {
                Eigen::Vector3d P(xy.x(), xy.y(), 0);
                P = c_T_t * P;
                img.push_back(cams[cam_idx].project(P));
            }
            obs.push_back({make_view(obj, img), btg, static_cast<int>(cam_idx)});
        }
    }
    return obs;
}

// RNG helpers
struct RNG final {
    std::mt19937 gen;
    explicit RNG(uint32_t seed=0xC001C0DE) : gen(seed) {}
    double uni(double a, double b) {
        std::uniform_real_distribution<double> d(a,b);
        return d(gen);
    }
    double gauss(double stddev) {
        std::normal_distribution<double> n(0.0, stddev);
        return n(gen);
    }
    Eigen::Vector3d rand_unit_axis() {
        // Marsaglia method
        double z = uni(-1.0, 1.0);
        double t = uni(0.0, 2.0 * std::numbers::pi);
        double r = std::sqrt(1.0 - z*z);
        return {r*std::cos(t), r*std::sin(t), z};
    }
};

struct SimulatedHandEye final {
    Eigen::Affine3d g_T_c_gt;  // ^gT_c
    Eigen::Affine3d b_T_t_gt;  // ^bT_t
    Camera<BrownConradyd> cam_gt;

    std::vector<Eigen::Affine3d> c_T_t;   // ^cT_t per frame
    std::vector<Eigen::Vector3d> obj_pts; // target points in t-frame
    std::vector<BundleObservation> observations;  // per frame {view, b_T_g, cam_idx}

    std::vector<Eigen::Affine3d> b_T_g() const {
        std::vector<Eigen::Affine3d> out; out.reserve(observations.size());
        for (const auto& obs : observations) out.push_back(obs.b_T_g);
        return out;
    }

    void make_sequence(size_t n_frames, RNG& rng) {
        c_T_t.clear();
        observations.clear();
        observations.reserve(n_frames);

        Eigen::Affine3d T = Eigen::Affine3d::Identity(); // ^bT_g at k=0
        for (size_t k = 0; k < n_frames; ++k) {
            observations.push_back({ make_view({}, {}), T, 0 });
            c_T_t.push_back( g_T_c_gt.inverse() * T.inverse() * b_T_t_gt );
            if (k + 1 < n_frames) {
                const double ang = deg2rad(rng.uni(5.0, 25.0));
                const Eigen::Vector3d ax = rng.rand_unit_axis();
                const Eigen::Vector3d dt( rng.uni(-0.10, 0.10),
                                          rng.uni(-0.10, 0.10),
                                          rng.uni(-0.10, 0.10) );
                const Eigen::Affine3d d = make_pose(dt, ax, ang);
                T = T * d; // cumulative
            }
        }
    }

    void make_target_grid(int rows, int cols, double spacing) {
        obj_pts.clear(); obj_pts.reserve(rows*cols);
        const double x0 = -0.5*(cols-1)*spacing;
        const double y0 = -0.5*(rows-1)*spacing;
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                obj_pts.emplace_back(x0 + c*spacing, y0 + r*spacing, 0.0);
    }

    void render_pixels(double noise_px = 0.0, RNG* rng = nullptr) {
        if (observations.size() != c_T_t.size()) observations.resize(c_T_t.size());
        for (size_t k=0; k<observations.size(); ++k) {
            auto& obs = observations[k];
            obs.view.clear();
            obs.view.reserve(obj_pts.size());
            const auto& Tct = c_T_t[k];
            for (const auto& P : obj_pts) {
                const Eigen::Vector3d Pc = Tct.linear()*P + Tct.translation();
                if (Pc.z() <= 1e-6) continue; // optional cull
                Eigen::Vector2d uv = cam_gt.project(Pc);
                if (noise_px > 0.0 && rng) { uv.x() += rng->gauss(noise_px); uv.y() += rng->gauss(noise_px); }
                obs.view.push_back({ {P.x(), P.y()}, {uv.x(), uv.y()} });
            }
        }
    }
};
