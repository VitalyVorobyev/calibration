
#include "calibration/homography.h"

// std
#include <array>
#include <thread>

// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace vitavision {

using Vec2 = Eigen::Vector2d;
using Mat3 = Eigen::Matrix3d;
using Params = std::array<double, 8>;

static Mat3 params_to_h(const Params& h) {
    Mat3 H;
    H << h[0], h[1], h[2],
         h[3], h[4], h[5],
         h[6], h[7], 1.0;
    return H;
}

// Normalize points (Hartley): translate to centroid, scale to mean distance sqrt(2).
static void normalize_points(const std::vector<Vec2>& pts,
                             std::vector<Vec2>& norm_pts,
                             Mat3& T)
{
    norm_pts.resize(pts.size());
    if (pts.empty()) {
        T.setIdentity();
        return;
    }

    // Centroid
    double cx = 0, cy = 0;
    for (const auto& p : pts) { cx += p.x(); cy += p.y(); }
    cx /= pts.size(); cy /= pts.size();

    // Mean distance to origin
    double mean_dist = 0.0;
    for (const auto& p : pts) {
        double dx = p.x() - cx, dy = p.y() - cy;
        mean_dist += std::sqrt(dx*dx + dy*dy);
    }
    mean_dist /= pts.size();
    double s = (mean_dist > 1e-12) ? std::sqrt(2.0) / mean_dist : 1.0;

    // Similarity transform
    T << s, 0, -s*cx,
         0, s, -s*cy,
         0, 0, 1;

    // Apply
    for (size_t i = 0; i < pts.size(); ++i) {
        Eigen::Vector3d ph(pts[i].x(), pts[i].y(), 1.0);
        Eigen::Vector3d q = T * ph;
        norm_pts[i] = Vec2(q.x()/q.z(), q.y()/q.z());
    }
}

// DLT initial estimate with normalization
Mat3 estimate_homography_dlt(const std::vector<Vec2>& src,
                             const std::vector<Vec2>& dst)
{
    const int N = static_cast<int>(src.size());
    std::vector<Vec2> src_n, dst_n;
    Mat3 T_src, T_dst;
    normalize_points(src, src_n, T_src);
    normalize_points(dst, dst_n, T_dst);

    // Build A (2N x 9)
    Eigen::MatrixXd A(2*N, 9);
    for (int i = 0; i < N; ++i) {
        const double x  = src_n[i].x();
        const double y  = src_n[i].y();
        const double u  = dst_n[i].x();
        const double v  = dst_n[i].y();

        // Row 2i
        A(2*i, 0) = 0.0;
        A(2*i, 1) = 0.0;
        A(2*i, 2) = 0.0;
        A(2*i, 3) = -x;
        A(2*i, 4) = -y;
        A(2*i, 5) = -1.0;
        A(2*i, 6) = v*x;
        A(2*i, 7) = v*y;
        A(2*i, 8) = v;

        // Row 2i+1
        A(2*i+1, 0) = x;
        A(2*i+1, 1) = y;
        A(2*i+1, 2) = 1.0;
        A(2*i+1, 3) = 0.0;
        A(2*i+1, 4) = 0.0;
        A(2*i+1, 5) = 0.0;
        A(2*i+1, 6) = -u*x;
        A(2*i+1, 7) = -u*y;
        A(2*i+1, 8) = -u;
    }

    // Solve Ah = 0, h = last singular vector
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd h = svd.matrixV().col(8);
    Mat3 Hn;
    Hn << h(0), h(1), h(2),
          h(3), h(4), h(5),
          h(6), h(7), h(8);

    // Denormalize: H = T_dst^{-1} * Hn * T_src
    Mat3 H = T_dst.inverse() * Hn * T_src;

    // Fix scale
    if (std::abs(H(2, 2)) > 1e-15) H /= H(2, 2);
    else                           H /= (H.norm() + 1e-15);

    return H;
}

// Ceres residual: maps (x,y) -> (u,v) using H(h) and compares to (u*, v*)
struct HomographyResidual {
    HomographyResidual(double x, double y, double u, double v)
        : x_(x), y_(y), u_(u), v_(v) {}

    template <typename T>
    bool operator()(const T* const h, T* residuals) const {
        // h[0..7]: 8 params, H22 = 1
        T X = T(x_), Y = T(y_);
        T den = h[6]*X + h[7]*Y + T(1);
        // Guard against division by zero in autodiff context
        if (ceres::isnan(den) || ceres::abs(den) < T(1e-15)) {
            residuals[0] = T(0);
            residuals[1] = T(0);
            return true;
        }
        T uu = (h[0]*X + h[1]*Y + h[2]) / den;
        T vv = (h[3]*X + h[4]*Y + h[5]) / den;
        residuals[0] = uu - T(u_);
        residuals[1] = vv - T(v_);
        return true;
    }

    static ceres::CostFunction* Create(double x, double y, double u, double v) {
        return new ceres::AutoDiffCostFunction<HomographyResidual, 2, 8>(
            new HomographyResidual(x, y, u, v));
    }

    double x_, y_, u_, v_;
};

Mat3 fit_homography(const std::vector<Vec2>& src,
                    const std::vector<Vec2>& dst)
{
    if (src.size() < 4 || dst.size() < 4 || src.size() != dst.size()) {
        throw std::invalid_argument("At least 4 correspondences are required.");
    }

    // Initial estimate via (normalized) DLT
    Mat3 H0 = estimate_homography_dlt(src, dst);
    // Convert to 8-parameter vector with H22 fixed to 1
    Params h = {
        H0(0,0), H0(0,1), H0(0,2),
        H0(1,0), H0(1,1), H0(1,2),
        H0(2,0), H0(2,1)
    };

    ceres::Problem problem;
    // Robust loss helps when outliers exist.
    ceres::LossFunction* loss = new ceres::HuberLoss(1.0);

    for (size_t i = 0; i < src.size(); ++i) {
        ceres::CostFunction* cost =
            HomographyResidual::Create(src[i].x(), src[i].y(), dst[i].x(), dst[i].y());
        problem.AddResidualBlock(cost, loss, h.data());
    }

    // (Optional) You can constrain parameters if needed:
    // e.g., none here. If you suspect degeneracy, consider parameterization tricks.

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 100;
    options.num_threads = std::max(1u, std::thread::hardware_concurrency());
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Reconstruct H and normalize scale
    Mat3 H = params_to_h(h);
    if (std::abs(H(2, 2)) > 1e-15) H /= H(2, 2);

    return H;
}

}  // namespace vitavision
