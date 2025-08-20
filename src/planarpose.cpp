#include "calibration/planarpose.h"

// std
#include <algorithm>
#include <numeric>

// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "calibration/homography.h"
#include "calibration/distortion.h"

namespace vitavision {

// Decompose homography in normalized camera coords: H = [r1 r2 t]
Eigen::Affine3d pose_from_homography_normalized(const Eigen::Matrix3d& H) {
    Eigen::Vector3d h1 = H.col(0);
    Eigen::Vector3d h2 = H.col(1);
    Eigen::Vector3d h3 = H.col(2);

    double s = std::sqrt(h1.norm() * h2.norm());
    if (s < 1e-12) s = 1.0;
    Eigen::Vector3d r1 = h1 / s;
    Eigen::Vector3d r2 = h2 / s;
    Eigen::Vector3d r3 = r1.cross(r2);

    // Orthonormalize to the nearest rotation
    Eigen::Matrix3d Rinit;
    Rinit.col(0) = r1;
    Rinit.col(1) = r2;
    Rinit.col(2) = r3;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(Rinit, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
    if (R.determinant() < 0) {
        Eigen::Matrix3d V = svd.matrixV();
        V.col(2) *= -1.0;
        R = svd.matrixU() * V.transpose();
    }
    Eigen::Vector3d t = h3 / s;
    if (R(2, 2) < 0) { // enforce cheirality (Z forward)
        R = -R; t = -t;
    }

    auto pose = Eigen::Affine3d::Identity();
    pose.linear() = R;
    pose.translation() = t;
    return pose;
}

// Convenience: one-shot planar pose from pixels & K
// Returns true on success; outputs R (world->cam) and t
Eigen::Affine3d estimate_planar_pose_dlt(const std::vector<Eigen::Vector2d>& obj_xy,
                                         const std::vector<Eigen::Vector2d>& img_uv,
                                         const Intrinsic& intrinsics) {
    if (obj_xy.size() < 4 || obj_xy.size() != img_uv.size()) {
        return Eigen::Affine3d::Identity();
    }
    
    std::vector<Eigen::Vector2d> img_norm(img_uv.size());
    std::transform(img_uv.begin(), img_uv.end(), img_norm.begin(),
        [&intrinsics](const Eigen::Vector2d& pix) {
            return intrinsics.pixel_to_norm(pix);
        });

    Eigen::Matrix3d H = estimate_homography_dlt(obj_xy, img_norm);
    return pose_from_homography_normalized(H);
}

struct PlanarObs {
    Eigen::Vector2d XY;   // target coords (Z=0)
    Eigen::Vector2d uv;   // observed pixel coords
};

using Pose6 = Eigen::Matrix<double, 6, 1>;

class PlanarPoseVPResidual : public ceres::CostFunction {
    std::vector<PlanarObs> obs_;
    double K_[4]; // fx, fy, cx, cy
    int num_radial_;

    // Build residuals = A*alpha* - b  with alpha* solved from LS for current pose
    void computeResiduals(const Pose6& pose6, double* residuals) const;

    // Convert (XY, pose) -> normalized (x,y) + observed (u,v)
    std::vector<Observation> buildObs(const Pose6& pose6) const;

public:
    PlanarPoseVPResidual(std::vector<PlanarObs> obs,
                         int num_radial,
                         const Intrinsic& intrinsics);

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override;
    
    // Solve distortion for a given pose (current intrinsics are fixed)
    Eigen::VectorXd SolveDistortionFor(const Pose6& pose6) const;
};

PlanarPoseVPResidual::PlanarPoseVPResidual(
    std::vector<PlanarObs> obs,
    int num_radial,
    const Intrinsic& intrinsics
) : obs_(std::move(obs)),
    K_{intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy},
    num_radial_(num_radial)
{
    set_num_residuals((int)obs_.size() * 2);
    // 6 params: angle-axis (3) + translation (3)
    mutable_parameter_block_sizes()->push_back(6);
}

void PlanarPoseVPResidual::computeResiduals(const Pose6& pose6, double* residuals) const {
    std::vector<Observation> o = buildObs(pose6);
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    LSDesign::build(o, num_radial_, K_[0], K_[1], K_[2], K_[3], A, b);
    Eigen::VectorXd alpha = LSDesign::solveNormal(A, b);
    Eigen::VectorXd r = A * alpha - b;
    Eigen::Map<Eigen::VectorXd>(residuals, r.size()) = r;
}

std::vector<Observation> PlanarPoseVPResidual::buildObs(const Pose6& pose6) const {
    const double* aa = pose6.data();              // angle-axis
    const double* t  = pose6.data() + 3;          // translation

    // TODO: make it class member to avoid heap allocations
    std::vector<Observation> o(obs_.size());
    std::transform(obs_.begin(), obs_.end(), o.begin(),
        [aa, t](const PlanarObs& s) {
            Eigen::Vector3d P(s.XY.x(), s.XY.y(), 0.0);
            Eigen::Vector3d Pc;
            ceres::AngleAxisRotatePoint(aa, P.data(), Pc.data());
            Pc += Eigen::Vector3d(t[0], t[1], t[2]);
            double invZ = 1.0 / Pc.z();
            Observation ob;
            ob.x = Pc.x() * invZ;
            ob.y = Pc.y() * invZ;
            ob.u = s.uv.x();
            ob.v = s.uv.y();
            return ob;
        });

    return o;
}

Eigen::VectorXd PlanarPoseVPResidual::SolveDistortionFor(const Pose6& pose6) const {
    std::vector<Observation> o = buildObs(pose6);
    return fit_distortion(o, K_[0], K_[1], K_[2], K_[3], num_radial_);
}

bool PlanarPoseVPResidual::Evaluate(double const* const* parameters,
                                    double* residuals,
                                    double** jacobians) const {
    const Pose6 pose6 = Pose6::Map(parameters[0]);
    computeResiduals(pose6, residuals);

    if (jacobians && jacobians[0]) {
        double* J = jacobians[0];
        const int m = num_residuals();
        std::vector<double> r_plus(m);
        std::vector<double> r_minus(m);

        Pose6 base = pose6;
        for (int k = 0; k < 6; ++k) {
            double step = (k < 3) ? 1e-6 : 1e-5; // rot in rad, trans in world units
            Pose6 p_plus = base;
            Pose6 p_minus = base;

            p_plus[k] += step;
            p_minus[k] -= step;

            computeResiduals(p_plus, r_plus.data());
            computeResiduals(p_minus, r_minus.data());
            for (int i = 0; i < m; ++i) {
                J[i * 6 + k] = (r_plus[i] - r_minus[i]) / (2.0 * step);
            }
        }
    }
    return true;
}

static Eigen::Affine3d axisangle_to_pose(const Pose6& pose6) {
    Eigen::Matrix3d rotation_matrix;
    ceres::AngleAxisToRotationMatrix(pose6.head<3>().data(), rotation_matrix.data());
    
    Eigen::Affine3d transform = Eigen::Affine3d::Identity();
    transform.linear() = rotation_matrix;
    transform.translation() = pose6.tail<3>();
    
    return transform;
}

PlanarPoseFitResult optimize_planar_pose(
    const std::vector<Eigen::Vector2d>& obj_xy,
    const std::vector<Eigen::Vector2d>& img_uv,
    const Intrinsic& intrinsics,
    int num_radial,
    bool verbose
) {
    PlanarPoseFitResult result;

    // Step 1: Estimate initial pose using DLT
    auto init_pose = estimate_planar_pose_dlt(obj_xy, img_uv, intrinsics);

    // Step 2: Optimize pose using non-linear least squares
    Pose6 pose6;
    ceres::RotationMatrixToAngleAxis(reinterpret_cast<const double*>(init_pose.rotation().data()), pose6.data());
    pose6[3] = init_pose.translation().x();
    pose6[4] = init_pose.translation().y();
    pose6[5] = init_pose.translation().z();

    std::vector<PlanarObs> view(obj_xy.size());
    std::transform(obj_xy.begin(), obj_xy.end(), img_uv.begin(), view.begin(),
        [](const Eigen::Vector2d& xy, const Eigen::Vector2d& uv) {
            return PlanarObs{xy, uv};
        });

    ceres::Problem p;
    auto* cost = new PlanarPoseVPResidual(view, num_radial, intrinsics);
    p.AddResidualBlock(cost, /*loss=*/nullptr, pose6.data());

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.minimizer_progress_to_stdout = verbose;
    opts.function_tolerance = 1e-12;
    opts.gradient_tolerance = 1e-12;
    opts.parameter_tolerance = 1e-12;

    ceres::Solver::Summary sum;
    ceres::Solve(opts, &p, &sum);
    result.summary = sum.BriefReport();

    // Best-fit distortion for the refined pose (if you want it)
    result.distortion = cost->SolveDistortionFor(pose6); // [k1..kK, p1, p2]

    // Residual stats & covariance (6x6 on pose)
    const int m = static_cast<int>(view.size()) * 2;
    std::vector<double> r(m);

    const double* parameter_blocks[] = {pose6.data()};
    cost->Evaluate(parameter_blocks, r.data(), nullptr);

    double ssr = 0.0;
    for (double e : r) ssr += e*e;
    const int dof = std::max(1, m - 6);
    const double sigma2 = ssr / dof;
    result.reprojection_error = std::sqrt(ssr / m);

    // Covariance block on pose
    ceres::Covariance::Options copt;
    ceres::Covariance cov(copt);
    std::vector<std::pair<const double*, const double*>> blocks = { {pose6.data(), pose6.data()} };
    if (!cov.Compute(blocks, &p)) {
        std::cerr << "Covariance computation failed.\n";
        return result;
    }

    double Cov6x6[36];
    cov.GetCovarianceBlock(pose6.data(), pose6.data(), Cov6x6);

    // Scale by residual variance (unit weights)
    Eigen::Map<Eigen::Matrix<double, 6, 6>> Cpose(Cov6x6);
    Cpose *= sigma2;
    result.covariance = Cpose;
    result.pose = axisangle_to_pose(pose6);

    return result;
}

}  // namespace vitavision
