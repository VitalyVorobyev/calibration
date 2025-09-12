/** @brief Helper structure to produce synthetic data */

#pragma once

// std
#include <random>
#include <numbers>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "calib/bundle.h"
#include "calib/pinhole.h"

#include "utils.h"

using namespace calib;

// -----------------------------------------------
// RNG helpers (deterministic, reusable, efficient)
// -----------------------------------------------
struct RNG final {
    using Engine = std::mt19937_64;
    Engine gen;

    explicit RNG(uint64_t seed = 0xC001C0DEu) : gen(seed) {}

    // U[0,1)
    double uni01() { return std::generate_canonical<double, 64>(gen); }

    // U[a,b]
    double uni(double a, double b) {
        std::uniform_real_distribution<double> d(a, b);
        return d(gen);
    }

    // N(0, stddev^2)
    double gauss(double stddev) {
        std::normal_distribution<double> n(0.0, stddev);
        return n(gen);
    }

    // Uniform on S^2 using normalized normals (no branch on sin/cos)
    Eigen::Vector3d rand_unit_axis() {
        std::normal_distribution<double> n(0.0, 1.0);
        Eigen::Vector3d v{n(gen), n(gen), n(gen)};
        const double s = v.norm();
        // Extremely unlikely, but guard anyway
        return (s > 0) ? (v / s) : Eigen::Vector3d(Eigen::Vector3d::UnitZ());
    }

    // Uniform angle in [-max_angle, max_angle] around a uniform axis
    Eigen::Quaterniond rand_quat(double max_angle_rad) {
        const Eigen::Vector3d ax = rand_unit_axis();
        const double ang = uni(-max_angle_rad, +max_angle_rad);
        return Eigen::Quaterniond(Eigen::AngleAxisd(ang, ax));
    }

    Eigen::Matrix3d rand_so3(double max_angle_rad) { return rand_quat(max_angle_rad).toRotationMatrix(); }

    // Translation inside an axis-aligned box
    Eigen::Vector3d rand_translation(const Eigen::Vector3d& lo, const Eigen::Vector3d& hi) {
        return { uni(lo.x(), hi.x()), uni(lo.y(), hi.y()), uni(lo.z(), hi.z()) };
    }

    // Small screw motion (rotation + translation)
    Eigen::Isometry3d rand_pose(double max_angle_rad,
                                const Eigen::Vector3d& t_lo,
                                const Eigen::Vector3d& t_hi) {
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.linear() = rand_so3(max_angle_rad);
        T.translation() = rand_translation(t_lo, t_hi);
        return T;
    }
};

struct SimulatedHandEye final {
    Eigen::Isometry3d g_se3_c_gt;  // ^gT_c  (gripper to camera)
    Eigen::Isometry3d b_se3_t_gt;  // ^bT_t  (base to target)
    PinholeCamera<BrownConradyd> cam_gt;

    std::vector<Eigen::Isometry3d> c_se3_t;   // ^cT_t per frame
    std::vector<Eigen::Vector3d>   obj_pts;   // target points in t-frame
    std::vector<BundleObservation> observations;  // per frame {view, b_se3_g, cam_idx}

    std::vector<Eigen::Isometry3d> b_se3_g() const {
        std::vector<Eigen::Isometry3d> out; out.reserve(observations.size());
        for (const auto& obs : observations) out.push_back(obs.b_se3_g);
        return out;
    }

    // Generate a well-conditioned motion sequence.
    //  - n_frames:   number of frames
    //  - rot_deg:    per-step rotation range [min,max] in degrees
    //  - trans_box:  per-step translation sampled in box [lo,hi] (meters)
    //  - start_Tbg:  optional starting ^bT_g (defaults to Identity)
    void make_sequence(size_t n_frames,
                       RNG& rng,
                       std::pair<double,double> rot_deg = {5.0, 25.0},
                       const Eigen::Vector3d& t_lo = {-0.10, -0.10, -0.10},
                       const Eigen::Vector3d& t_hi = {+0.10, +0.10, +0.10},
                       const std::optional<Eigen::Isometry3d>& start_b_se3_g = std::nullopt) {
        c_se3_t.clear();
        observations.clear();
        c_se3_t.reserve(n_frames);
        observations.reserve(n_frames);

        Eigen::Isometry3d b_se3_g = start_b_se3_g.value_or(Eigen::Isometry3d::Identity()); // ^bT_g at k=0

        for (size_t k = 0; k < n_frames; ++k) {
            // Record observation *first* to keep vectors aligned
            observations.push_back({ make_view({}, {}), b_se3_g, 0 });

            // Compute ^cT_t = (^gT_c)^-1 * (^bT_g)^-1 * (^bT_t)
            c_se3_t.push_back( g_se3_c_gt.inverse() * b_se3_g.inverse() * b_se3_t_gt );

            if (k + 1 < n_frames) {
                const double ang = deg2rad(rng.uni(rot_deg.first, rot_deg.second));
                const Eigen::Vector3d ax = rng.rand_unit_axis();
                const Eigen::Vector3d dt = rng.rand_translation(t_lo, t_hi);
                const Eigen::Isometry3d d = make_pose(dt, ax, ang);
                b_se3_g = b_se3_g * d; // cumulative screw motion

                // Optional: re-center to avoid drifting absurdly far
                // (keeps conditioning OK without changing relative motions)
                if (k % 8 == 7) {
                    Eigen::Isometry3d recenter = Eigen::Isometry3d::Identity();
                    recenter.translation() = -b_se3_g.translation() * 0.2; // gentle pullback
                    b_se3_g = recenter * b_se3_g;
                }
            }
        }

        // Invariants
        assert(observations.size() == c_se3_t.size() && "sequence vectors must stay aligned");
    }

    // (rows x cols) planar grid centered at origin in t-frame, spacing in meters
    void make_target_grid(int rows, int cols, double spacing) {
        obj_pts.clear();
        obj_pts.reserve(static_cast<size_t>(rows) * static_cast<size_t>(cols));

        const double x0 = -0.5 * (cols - 1) * spacing;
        const double y0 = -0.5 * (rows - 1) * spacing;

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                obj_pts.emplace_back(x0 + c * spacing, y0 + r * spacing, 0.0);
            }
        }
    }

    // Render observations:
    //  - noise_px:       Gaussian pixel noise (stddev)
    //  - rng:            RNG for noise/outliers/dropouts (optional)
    //  - img_w/h:        optional bounds to keep only in-image points
    //  - dropout_prob:   probability to drop a visible point (simulates detection failures)
    //  - outlier_prob:   probability to replace a point with a random pixel (simulates mismatches)
    //  - z_near:         minimal positive depth to accept
    void render_pixels(double noise_px = 0.0,
                       RNG* rng = nullptr,
                       std::optional<int> img_w = std::nullopt,
                       std::optional<int> img_h = std::nullopt,
                       double dropout_prob = 0.0,
                       double outlier_prob = 0.0,
                       double z_near = 1e-6) {
        // Never silently resize; keep the 1:1 mapping created in make_sequence.
        assert(observations.size() == c_se3_t.size()
               && "Call make_sequence() first; vectors must be aligned");

        const bool check_bounds = img_w.has_value() && img_h.has_value();

        for (size_t k = 0; k < observations.size(); ++k) {
            auto& obs = observations[k];
            obs.view.clear();
            obs.view.reserve(obj_pts.size());

            const auto& Tct = c_se3_t[k];

            for (const auto& Pt : obj_pts) {
                const Eigen::Vector3d Pc = Tct * Pt.homogeneous();
                if (Pc.z() <= z_near) continue; // behind camera or too close

                Eigen::Vector2d uv = cam_gt.project(Pc);

                // Optional image bounds culling
                if (check_bounds) {
                    if (!(uv.x() >= 0 && uv.x() < *img_w && uv.y() >= 0 && uv.y() < *img_h))
                        continue;
                }

                // Optional dropout
                if (rng && dropout_prob > 0.0 && rng->uni01() < dropout_prob)
                    continue;

                // Optional outliers
                if (rng && outlier_prob > 0.0 && rng->uni01() < outlier_prob) {
                    if (check_bounds) {
                        uv.x() = rng->uni(0.0, static_cast<double>(*img_w - 1));
                        uv.y() = rng->uni(0.0, static_cast<double>(*img_h - 1));
                    } else {
                        uv.x() += rng->gauss(50.0);
                        uv.y() += rng->gauss(50.0);
                    }
                } else if (noise_px > 0.0 && rng) {
                    uv.x() += rng->gauss(noise_px);
                    uv.y() += rng->gauss(noise_px);
                }

                // Store (target XY) -> (pixel UV) pair
                obs.view.push_back({ {Pt.x(), Pt.y()}, {uv.x(), uv.y()} });
            }

            // Optional: if a frame becomes too sparse, it’s often better to drop it entirely
            // to avoid rank issues in homography estimation
            if (obs.view.size() < 12) { obs.view.clear(); /* mark unusable */ }
        }
    }
};
