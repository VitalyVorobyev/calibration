#include "calib/handeye.h"

// std
#include <iostream>
#include <numbers>

#include "observationutils.h"

namespace calib {

std::vector<MotionPair> build_all_pairs(
    const std::vector<Eigen::Affine3d>& b_T_g, const std::vector<Eigen::Affine3d>& c_T_t,
    double min_angle_deg,       // discard too-small motions
    bool reject_axis_parallel,  // guard against ill-conditioning
    double axis_parallel_eps) {
    if (b_T_g.size() < 2 || b_T_g.size() != c_T_t.size()) {
        throw std::runtime_error("Inconsistent hand-eye input sizes");
    }
    const size_t n = b_T_g.size();
    const double min_angle = min_angle_deg * std::numbers::pi / 180.0;

    std::vector<MotionPair> pairs;
    pairs.reserve(n * (n - 1) / 2);

    for (size_t i = 0; i + 1 < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            Eigen::Affine3d A = b_T_g[i].inverse() * b_T_g[j];
            Eigen::Affine3d B = c_T_t[i] * c_T_t[j].inverse();

            MotionPair mp;
            mp.RA = projectToSO3(A.linear());
            mp.RB = projectToSO3(B.linear());
            mp.tA = A.translation();
            mp.tB = B.translation();

            Eigen::Vector3d alpha = logSO3(mp.RA);
            Eigen::Vector3d beta = logSO3(mp.RB);
            const double a = alpha.norm();
            const double b = beta.norm();

            // weight by the smaller rotation magnitude; skip if too small
            const double w = std::min(a, b);
            if (w < min_angle) {
                std::cerr << "Skipping pair (" << i << "," << j
                          << ") with too small motion: " << (w * 180.0 / std::numbers::pi)
                          << " deg\n";
                continue;
            }

            if (reject_axis_parallel) {
                const double aa = (a < 1e-9) ? 0.0 : 1.0;
                const double bb = (b < 1e-9) ? 0.0 : 1.0;
                if (aa * bb > 0.0) {
                    double sin_axis = (alpha.normalized().cross(beta.normalized())).norm();
                    if (sin_axis < axis_parallel_eps) {
                        std::cerr << "Skipping pair (" << i << "," << j
                                  << ") with near-parallel axes\n";
                        continue;  // nearly same axis
                    }
                }
            }

#if 0
            mp.sqrt_weight = std::sqrt(w);
#endif

            pairs.push_back(std::move(mp));
        }
    }
    if (pairs.empty()) {
        throw std::runtime_error(
            "No valid motion pairs after filtering. Increase motion or relax thresholds.");
    }
    return pairs;
}

// ---------- weighted Tsai–Lenz rotation over all pairs ----------
static Eigen::Matrix3d estimate_rotation_allpairs_weighted(const std::vector<MotionPair>& pairs) {
    const int m = static_cast<int>(pairs.size());
    Eigen::MatrixXd M(3 * m, 3);
    Eigen::VectorXd d(3 * m);

    for (int k = 0; k < m; ++k) {
        Eigen::Vector3d alpha = logSO3(pairs[k].RA);
        Eigen::Vector3d beta = logSO3(pairs[k].RB);
        constexpr double s = 1;  // pairs[k].sqrt_weight;

        M.block<3, 3>(static_cast<Eigen::Index>(3 * k), 0) = s * skew(alpha + beta);
        d.segment<3>(static_cast<Eigen::Index>(3 * k)) = s * (beta - alpha);
    }

    Eigen::Vector3d r = ridge_llsq(M, d, 1e-12);
    return expSO3(r);
}

// ---------- weighted Tsai–Lenz translation over all pairs ----------
static Eigen::Vector3d estimate_translation_allpairs_weighted(const std::vector<MotionPair>& pairs,
                                                              const Eigen::Matrix3d& RX) {
    const int m = static_cast<int>(pairs.size());
    Eigen::MatrixXd C(3 * m, 3);
    Eigen::VectorXd w(3 * m);

    for (int k = 0; k < m; ++k) {
        const Eigen::Matrix3d& RA = pairs[k].RA;
        const Eigen::Vector3d& tA = pairs[k].tA;
        const Eigen::Vector3d& tB = pairs[k].tB;
        constexpr double s = 1;  // pairs[k].sqrt_weight;

        C.block<3, 3>(static_cast<Eigen::Index>(3 * k), 0) = s * (RA - Eigen::Matrix3d::Identity());
        w.segment<3>(static_cast<Eigen::Index>(3 * k)) = s * (RX * tB - tA);
    }

    return ridge_llsq(C, w, 1e-12);
}

// ---------- public API: linear init + non-linear refine ----------
Eigen::Affine3d estimate_handeye_dlt(const std::vector<Eigen::Affine3d>& base_T_gripper,
                                     const std::vector<Eigen::Affine3d>& camera_T_target,
                                     double min_angle_deg) {
    auto pairs = build_all_pairs(base_T_gripper, camera_T_target, min_angle_deg);
    const Eigen::Matrix3d RX = estimate_rotation_allpairs_weighted(pairs);
    const Eigen::Vector3d g_t_c = estimate_translation_allpairs_weighted(pairs, RX);

    Eigen::Affine3d X = Eigen::Affine3d::Identity();
    X.linear() = RX;
    X.translation() = g_t_c;
    return X;
}

}  // namespace calib
