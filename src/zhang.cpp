#include "zhang.h"

// std
#include <iostream>

namespace calib {

/** @brief Constructs B = K^{-T} K^{-1} from the 6-vector b */
static auto zhang_bmtx(const Eigen::VectorXd& b) -> Eigen::Matrix3d {
    Eigen::Matrix3d bmtx;
    bmtx << b(0), b(1), b(3), b(1), b(2), b(4), b(3), b(4), b(5);
    return 0.5 * (bmtx + bmtx.transpose());  // symmetrize
}

static void check_conic_decomposition(const Eigen::Matrix3d& bmtx, const Eigen::Matrix3d& kmtx) {
    // Optional self-check: B ≈ K^{-T} K^{-1} up to scale (should be exact after our normalization)
    const Eigen::Matrix3d bhat = kmtx.inverse().transpose() * kmtx.inverse();
    if (bhat.allFinite()) {
        // Compare relative Frobenius error after aligning scale by a single element
        double s = 1.0;
        if (std::abs(bhat(2, 2)) > 0) s = bmtx(2, 2) / bhat(2, 2);
        const double rel_err = (bmtx - s * bhat).norm() / std::max(1e-12, bmtx.norm());
        if (!std::isfinite(rel_err) || rel_err > 1e-6) {
            std::cerr << "Zhang: B consistency warning, rel_err=" << rel_err << "\n";
        }
    }
}

// b is homogeneous: recover K from B = K^{-T} K^{-1} via Cholesky of B
static auto kmtx_from_dual_conic(const Eigen::VectorXd& bv) -> std::optional<Eigen::Matrix3d> {
    // Expect a 6-vector: [b11, b12, b22, b13, b23, b33]^T
    if (bv.size() != 6) {
        std::cerr << "Zhang: dual-conic vector must have size 6, got " << bv.size() << "\n";
        return std::nullopt;
    }

    const auto try_factor = [](const Eigen::Matrix3d& bmtx) -> std::optional<Eigen::Matrix3d> {
        // B should be symmetric positive-definite (omega = K^{-T} K^{-1})
        if (!bmtx.allFinite()) return std::nullopt;

        // Fast SPD check via LLT; also gives us the factor in one shot
        Eigen::LLT<Eigen::Matrix3d> llt(bmtx);
        if (llt.info() != Eigen::Success) return std::nullopt;

        // Cholesky: B = U^T * U, with U upper-triangular, diag > 0
        const Eigen::Matrix3d umtx = llt.matrixU();

        // Since B = (K^{-1})^T (K^{-1}), we have U = K^{-1}  (same triangular form)
        Eigen::Matrix3d kmtx = umtx.inverse();
        if (!kmtx.allFinite()) return std::nullopt;

        // Normalize so K(2,2) = 1
        const double k22 = kmtx(2, 2);
        if (std::abs(k22) < 1e-15) return std::nullopt;
        kmtx /= k22;

        // Ensure a conventional calibration matrix: positive focal lengths
        if (kmtx(0, 0) <= 0.0 || kmtx(1, 1) <= 0.0) {
            // Flip sign uniformly if needed (numerical edge cases)
            kmtx = -kmtx;
        }

        check_conic_decomposition(bmtx, kmtx);
        return kmtx;
    };

    // Rebuild symmetric B from the 6-vector
    const Eigen::Matrix3d bmtx = zhang_bmtx(bv);

    // Try as-is, then the opposite sign (b is homogeneous)
    if (auto kmtx = try_factor(bmtx))  return kmtx;
    if (auto kmtx = try_factor(-bmtx)) return kmtx;

    std::cerr << "Zhang: failed to recover K from dual conic (both signs).\n";
    return std::nullopt;
}

// ---------- Zhang: recover K from homographies ----------
static auto v_ij(const Eigen::Matrix3d& hmtx, int i, int j) -> Eigen::Matrix<double, 1, 6> {
    assert(0 <= i && i < 3 && 0 <= j && j < 3);
    const double h0i = hmtx(0, i);
    const double h1i = hmtx(1, i);
    const double h2i = hmtx(2, i);
    const double h0j = hmtx(0, j);
    const double h1j = hmtx(1, j);
    const double h2j = hmtx(2, j);

    Eigen::Matrix<double, 1, 6> v;
    v <<
        h0i * h0j,
        h0i * h1j + h1i * h0j,
        h1i * h1j,
        h0i * h2j + h2i * h0j,
        h1i * h2j + h2i * h1j,
        h2i * h2j;
    return v;
}

static auto normalize_hmtx(const Eigen::Matrix3d& hmtx) -> Eigen::Matrix3d {
    Eigen::Matrix3d hnorm = hmtx;
#if 0
    const double n1 = hnorm.col(0).norm();
    const double n2 = hnorm.col(1).norm();
    if (n1 > 0) hnorm.col(0) /= n1;
    if (n2 > 0) hnorm.col(1) /= n2;
    // Consistent orientation
    if ((hnorm.col(0).cross(hnorm.col(1))).dot(hnorm.col(2)) < 0) hnorm = -hnorm;
#endif
    return hnorm;
}

static auto make_zhang_design_matrix(const std::vector<HomographyResult>& hs)
    -> std::optional<Eigen::MatrixXd> {
    const int m = static_cast<int>(hs.size());
    if (m < 4) {
        std::cerr << "Zhang method requires at least 4 views\n";
        return std::nullopt;
    }

    Eigen::MatrixXd vmtx(2 * m, 6);
    for (Eigen::Index k = 0; k < m; ++k) {
        Eigen::Matrix3d hmtx = normalize_hmtx(hs[static_cast<size_t>(k)].hmtx);  // <-- important
        Eigen::Matrix<double, 1, 6> v12 = v_ij(hmtx, 0, 1);
        Eigen::Matrix<double, 1, 6> v11 = v_ij(hmtx, 0, 0);
        Eigen::Matrix<double, 1, 6> v22 = v_ij(hmtx, 1, 1);
        vmtx.row(2 * k) = v12;
        vmtx.row(2 * k + 1) = v11 - v22;
    }
    return vmtx;
}

auto zhang_intrinsics_from_hs(const std::vector<HomographyResult>& hs)
    -> std::optional<CameraMatrix> {
    const auto vmtx_opt = make_zhang_design_matrix(hs);
    if (!vmtx_opt.has_value()) {
        std::cout << "Zhang design matrix creation failed.\n";
        return std::nullopt;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(vmtx_opt.value(), Eigen::ComputeFullV);

    // The null space should correspond to the smallest singular value
    Eigen::VectorXd bvec = svd.matrixV().col(5);  // smallest singular value
    Eigen::VectorXd resid = vmtx_opt.value() * bvec;
    double rms = std::sqrt(resid.squaredNorm() / static_cast<double>(resid.size()));
    if (rms > 1e-3) {
        std::cout << "Zhang warning: large residual in solving for b: " << rms << '\n';
    } else {
        std::cout << "Zhang: solved for b with residual " << rms << '\n';
    }

    // Try both signs of the b vector
    std::optional<Eigen::Matrix3d> kmtx_opt = kmtx_from_dual_conic(bvec);
    if (!kmtx_opt.has_value()) {
        std::cout << "Zhang kmtx_from_dual_conic failed for one sign, trying the other.\n";
        bvec *= -1;
        kmtx_opt = kmtx_from_dual_conic(bvec);
        if (!kmtx_opt.has_value()) {
            std::cout << "Zhang kmtx_from_dual_conic failed for both signs.\n";
            return std::nullopt;
        }
    }

    return CameraMatrix{.fx = kmtx_opt.value()(0, 0),
                        .fy = kmtx_opt.value()(1, 1),
                        .cx = kmtx_opt.value()(0, 2),
                        .cy = kmtx_opt.value()(1, 2),
                        .skew = kmtx_opt.value()(0, 1)};
}

}  // namespace calib
