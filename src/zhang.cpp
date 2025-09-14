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

// b is homogeneous: try both signs to make B^{-1} SPD
static auto kmtx_from_dual_conic(const Eigen::VectorXd& bv) -> std::optional<Eigen::Matrix3d> {
    Eigen::Matrix3d bmtx = zhang_bmtx(bv);
    if (!bmtx.allFinite()) {
        return std::nullopt;
    }

    // Check if B has the right structure for a dual conic
    // B should be symmetric and represent omega = K^(-T) * K^(-1)
    double det_upper = bmtx(0, 0) * bmtx(1, 1) - bmtx(0, 1) * bmtx(0, 1);
    if (det_upper <= 0) {
        std::cerr << "Zhang: invalid dual conic structure: " << det_upper << '\n';
        return std::nullopt;
    }

    // We need B^(-1) = K*K^T to be positive definite
    Eigen::Matrix3d binv = bmtx.inverse();
    if (!binv.allFinite()) {
        return std::nullopt;
    }

    // Check if Binv is positive definite using eigenvalues
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(binv);
    Eigen::Vector3d eigenvals = eigensolver.eigenvalues();

    if (eigensolver.info() != Eigen::Success) {
        return std::nullopt;
    }

    if (eigenvals.minCoeff() <= 1e-8) {
        std::cerr << "Zhang: Binv not positive definite\n";
        return std::nullopt;
    }

    // Use Cholesky decomposition to get K
    Eigen::LLT<Eigen::Matrix3d> llt(binv);
    if (llt.info() != Eigen::Success) {
        std::cerr << "Zhang: LLT decomposition failed\n";
        return std::nullopt;
    }

    // Upper-triangular K (positive diag), normalize K(2,2)=1
    Eigen::Matrix3d kmtx = llt.matrixU();
    if (kmtx(0, 0) <= 0 || kmtx(1, 1) <= 0 || kmtx(2, 2) <= 0) {
        std::cerr << "Zhang: invalid K diagonal\n";
        return std::nullopt;
    }
    kmtx /= kmtx(2, 2);
    if (!kmtx.allFinite()) {
        std::cerr << "Zhang: invalid K matrix\n";
        return std::nullopt;
    }
    return kmtx;
};

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
    v << h0i * h0j, h0i * h1j + h1i * h0j, h1i * h1j, h0i * h2j + h2i * h0j, h1i * h2j + h2i * h1j,
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
