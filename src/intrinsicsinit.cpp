#include "calib/intrinsicsinit.h"

// std
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>

// eigen
#include <Eigen/Dense>

namespace calib {

// ---------- Zhang: recover K from homographies ----------

static inline Eigen::Matrix<double, 1, 6> v_ij(const Eigen::Matrix3d& H, int i, int j) {
    // H columns: h1, h2, h3
    // v_ij = [h1i h1j, h1i h2j + h2i h1j, h2i h2j, h3i h1j + h1i h3j, h3i h2j + h2i h3j, h3i h3j]
    const Eigen::Vector3d h1 = H.col(0);
    const Eigen::Vector3d h2 = H.col(1);
    const Eigen::Vector3d h3 = H.col(2);
    Eigen::Matrix<double, 1, 6> v;
    v << h1(i) * h1(j), h1(i) * h2(j) + h2(i) * h1(j), h2(i) * h2(j), h3(i) * h1(j) + h1(i) * h3(j),
        h3(i) * h2(j) + h2(i) * h3(j), h3(i) * h3(j);
    return v;
}

static bool zhang_intrinsics_from_hs(const std::vector<Eigen::Matrix3d>& Hs, Eigen::Matrix3d& K_out,
                                     double& fx, double& fy, double& cx, double& cy, double& skew) {
    const int m = static_cast<int>(Hs.size());
    if (m < 2) return false;  // need at least 2 views

    Eigen::MatrixXd V(2 * m, 6);
    for (int i = 0; i < m; ++i) {
        const auto& H = Hs[i];
        Eigen::Matrix<double, 1, 6> v12 = v_ij(H, 0, 1);
        Eigen::Matrix<double, 1, 6> v11 = v_ij(H, 0, 0);
        Eigen::Matrix<double, 1, 6> v22 = v_ij(H, 1, 1);
        V.row(2 * i) = v12;
        V.row(2 * i + 1) = v11 - v22;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(V, Eigen::ComputeFullV);
    Eigen::VectorXd b = svd.matrixV().col(5);  // smallest singular value

    // b = [B11, B12, B22, B13, B23, B33]^T
    double B11 = b(0), B12 = b(1), B22 = b(2), B13 = b(3), B23 = b(4), B33 = b(5);

    // Enforce sign so that fx, fy are real
    if (B11 < 0) {
        B11 *= -1;
        B12 *= -1;
        B22 *= -1;
        B13 *= -1;
        B23 *= -1;
        B33 *= -1;
    }

    const double v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 * B12);
    const double lambda = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11;
    if (!(lambda > 0)) return false;

    const double alpha = std::sqrt(lambda / B11);
    const double beta = std::sqrt(lambda * B11 / (B11 * B22 - B12 * B12));
    const double gamma = -B12 * alpha * alpha * beta / lambda;
    const double u0 = (gamma * v0 / alpha) - (B13 * alpha * alpha / lambda);

    fx = alpha;
    fy = beta;
    cx = u0;
    cy = v0;
    skew = gamma;

    K_out.setIdentity();
    K_out(0, 0) = fx;
    K_out(0, 1) = skew;
    K_out(0, 2) = cx;
    K_out(1, 1) = fy;
    K_out(1, 2) = cy;
    return true;
}

// ---------- Extrinsics per view from H and K ----------

static bool extrinsics_from_H(const Eigen::Matrix3d& K, const Eigen::Matrix3d& H,
                              Eigen::Matrix3d& R, Eigen::Vector3d& t) {
    Eigen::Matrix3d Kinv = K.inverse();
    Eigen::Vector3d h1 = H.col(0);
    Eigen::Vector3d h2 = H.col(1);
    Eigen::Vector3d h3 = H.col(2);

    Eigen::Vector3d r1 = Kinv * h1;
    Eigen::Vector3d r2 = Kinv * h2;
    double s = 1.0 / r1.norm();
    r1 *= s;
    r2 *= s;
    Eigen::Vector3d r3 = r1.cross(r2);
    t = s * (Kinv * h3);

    // Orthonormalize
    Eigen::Matrix3d Rtmp;
    Rtmp.col(0) = r1;
    Rtmp.col(1) = r2;
    Rtmp.col(2) = r3;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(Rtmp, Eigen::ComputeFullU | Eigen::ComputeFullV);
    R = svd.matrixU() * svd.matrixV().transpose();
    if (R.determinant() < 0) {
        R.col(2) *= -1.0;  // fix reflection
    }

    return true;
}

// ---------- Distortion init (linear LS for k1,k2,p1,p2) ----------

static std::vector<double> init_distortion_brown(const Eigen::Matrix3d& K,
                                                 const std::vector<Eigen::Isometry3d>& c_T_t,
                                                 const std::vector<View2DPlanar>& views_inliers) {
    // We’ll solve for [k1,k2,p1,p2] in a single LS:
    // dx = x_u*r2*k1 + x_u*r4*k2 + 2*p1*x_u*y_u + p2*(r2 + 2*x_u^2)
    // dy = y_u*r2*k1 + y_u*r4*k2 + p1*(r2 + 2*y_u^2) + 2*p2*x_u*y_u
    // where (x_d, y_d) = K^{-1}[u v 1], (x_u, y_u) from ideal projection with (R,t).
    const int total_pts = [&] {
        int s = 0;
        for (auto& v : views_inliers) s += (int)v.pixels.size();
        return s;
    }();
    if (total_pts < 12) return {0, 0, 0, 0};

    Eigen::MatrixXd A(2 * total_pts, 4);
    Eigen::VectorXd b(2 * total_pts);

    Eigen::Matrix3d Kinv = K.inverse();

    int row = 0;
    for (size_t i = 0; i < views_inliers.size(); ++i) {
        const auto& view = views_inliers[i];
        const auto& T = c_T_t[i];  // camera_T_target
        const Eigen::Matrix3d R = T.rotation();
        const Eigen::Vector3d t = T.translation();

        for (size_t j = 0; j < view.pixels.size(); ++j) {
            // Ideal (undistorted) normalized projection of target point
            const double X = view.planar_xy[j].x();
            const double Y = view.planar_xy[j].y();
            Eigen::Vector3d Xc = R * Eigen::Vector3d(X, Y, 0.0) + t;
            const double x_u = Xc.x() / Xc.z();
            const double y_u = Xc.y() / Xc.z();

            // Measured normalized (distorted) coords
            Eigen::Vector3d ph(view.pixels[j].x(), view.pixels[j].y(), 1.0);
            Eigen::Vector3d pn = Kinv * ph;
            const double x_d = pn.x() / pn.z();
            const double y_d = pn.y() / pn.z();

            const double r2 = x_u * x_u + y_u * y_u;
            const double r4 = r2 * r2;

            // dx, dy
            const double dx = x_d - x_u;
            const double dy = y_d - y_u;

            // Row for dx: [x_u*r2, x_u*r4, 2*x_u*y_u, (r2 + 2*x_u^2)]
            A(row, 0) = x_u * r2;
            A(row, 1) = x_u * r4;
            A(row, 2) = 2.0 * x_u * y_u;
            A(row, 3) = (r2 + 2.0 * x_u * x_u);
            b(row) = dx;
            ++row;

            // Row for dy: [y_u*r2, y_u*r4, (r2 + 2*y_u^2), 2*x_u*y_u]
            A(row, 0) = y_u * r2;
            A(row, 1) = y_u * r4;
            A(row, 2) = (r2 + 2.0 * y_u * y_u);
            A(row, 3) = 2.0 * x_u * y_u;
            b(row) = dy;
            ++row;
        }
    }

    Eigen::Vector4d x = A.colPivHouseholderQr().solve(b);
    return {x(0), x(1), x(2), x(3)};  // k1,k2,p1,p2
}

// ---------- Public driver ----------

IntrinsicsInit initialize_from_planar(const std::vector<PlanarView>& views,
                                      const ImageSize& image_size, const InitOptions& opts) {
    IntrinsicsInit out;
    std::ostringstream log;

    // 0) Validate
    if (image_size.width <= 0 || image_size.height <= 0) {
        log << "Invalid image_size.\n";
        out.log = log.str();
        return out;
    }
    if (views.size() < 2) {
        log << "Need at least 2 views; got " << views.size() << ".\n";
        out.log = log.str();
        return out;
    }

    // 1) Per-view homographies (RANSAC + refit)
    std::vector<Eigen::Matrix3d> Hs;
    std::vector<PlanarView> views_after_inlier_filter;
    out.homographies.reserve(views.size());

    for (size_t i = 0; i < views.size(); ++i) {
        const auto& v = views[i];
        if (v.pixels.size() != v.planar_xy.size() || v.pixels.size() < 4) {
            log << "View " << i << ": invalid correspondences.\n";
            continue;
        }

        HomographyResult hr =
            opts.use_ransac ? ransac_homography(v.planar_xy, v.pixels, opts.ransac_max_iters,
                                                opts.ransac_thresh_px, opts.ransac_min_inliers)
                            : estimate_homography(v.planar_xy, v.pixels);

        if (!hr.success) {
            log << "View " << i << ": homography estimation failed.\n";
            continue;
        }

        // Optionally drop bad views
        if (opts.drop_bad_views && hr.symmetric_rms_px > opts.max_view_transfer_rms_px) {
            log << "View " << i << ": dropped (symmetric RMS=" << hr.symmetric_rms_px << " px).\n";
            continue;
        }

        // Keep only inliers for downstream stages
        View2DPlanar v_in;
        v_in.pixels.reserve(hr.inliers.size());
        v_in.planar_xy.reserve(hr.inliers.size());
        for (int id : hr.inliers) {
            v_in.pixels.push_back(v.pixels[id]);
            v_in.planar_xy.push_back(v.planar_xy[id]);
        }

        out.homographies.push_back(hr);
        Hs.push_back(hr.H);
        views_after_inlier_filter.push_back(std::move(v_in));
    }

    if (Hs.size() < 2) {
        log << "Too few valid homographies (" << Hs.size() << ").\n";
        out.log = log.str();
        return out;
    }

    // 2) Zhang: recover K
    if (!zhang_intrinsics_from_hs(Hs, out.K, out.fx, out.fy, out.cx, out.cy, out.skew)) {
        log << "Zhang closed-form intrinsics failed.\n";
        out.log = log.str();
        return out;
    }
    if (opts.assume_zero_skew) {
        out.skew = 0.0;
        out.K(0, 1) = 0.0;
    }
    log << "Init K:\n" << out.K << "\n";

    // 3) Per-view extrinsics from H
    out.c_T_t.resize(Hs.size());
    out.per_view_forward_rms_px.resize(Hs.size(), 0.0);

    for (size_t i = 0; i < Hs.size(); ++i) {
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        if (!extrinsics_from_H(out.K, Hs[i], R, t)) {
            log << "View " << i << ": extrinsics recovery failed.\n";
            continue;
        }
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.linear() = R;
        T.translation() = t;
        out.c_T_t[i] = T;

        // forward reprojection RMS (without distortion)
        const auto& v = views_after_inlier_filter[i];
        double ss = 0.0;
        for (size_t j = 0; j < v.pixels.size(); ++j) {
            Eigen::Vector3d Xc =
                R * Eigen::Vector3d(v.planar_xy[j].x(), v.planar_xy[j].y(), 0.0) + t;
            Eigen::Vector3d uvh = out.K * Eigen::Vector3d(Xc.x() / Xc.z(), Xc.y() / Xc.z(), 1.0);
            Eigen::Vector2d uv = uvh.hnormalized();
            ss += (uv - v.pixels[j]).squaredNorm();
        }
        out.per_view_forward_rms_px[i] = std::sqrt(ss / std::max<size_t>(1, v.pixels.size()));
    }

    // 4) Distortion init (linear LS)
    out.dist = init_distortion_brown(out.K, out.c_T_t, views_after_inlier_filter);

    log << "Distortion init [k1 k2 p1 p2] = " << out.dist[0] << " " << out.dist[1] << " "
        << out.dist[2] << " " << out.dist[3] << "\n";

    out.success = true;
    out.log = log.str();
    return out;
}

}  // namespace calib
