#include "calib/intrinsics.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "calib/distortion.h"
#include "calib/homography.h"
#include "calib/scheimpflug.h"
#include "ceresutils.h"
#include "observationutils.h"
#include "residuals/intrinsicresidual.h"

namespace calib {

template <camera_model CameraT>
struct IntrinsicBlocks final : public ProblemParamBlocks {
    static constexpr size_t intr_size = CameraTraits<CameraT>::param_count;
    std::vector<std::array<double, 4>> c_quat_t;
    std::vector<std::array<double, 3>> c_tra_t;
    std::array<double, intr_size> intr{};

    explicit IntrinsicBlocks(size_t numviews) : c_quat_t(numviews), c_tra_t(numviews) {}

    static IntrinsicBlocks create(const CameraT& camera,
                                  const std::vector<Eigen::Isometry3d>& init_c_se3_t) {
        const size_t num_views = init_c_se3_t.size();
        IntrinsicBlocks blocks(num_views);

        CameraTraits<CameraT>::to_array(camera, blocks.intr);
        for (size_t v = 0; v < num_views; ++v) {
            populate_quat_tran(init_c_se3_t[v], blocks.c_quat_t[v], blocks.c_tra_t[v]);
        }
        return blocks;
    }

    [[nodiscard]] std::vector<ParamBlock> get_param_blocks() const override {
        std::vector<ParamBlock> blocks;
        blocks.emplace_back(intr.data(), intr.size(), intr_size);

        // Reserve space for efficiency
        blocks.reserve(1 + c_quat_t.size() + c_tra_t.size());

        // Add quaternion blocks using std::transform
        std::transform(c_quat_t.begin(), c_quat_t.end(), std::back_inserter(blocks),
                       [](const auto& i) { return ParamBlock{i.data(), i.size(), 3}; });

        // Add translation blocks using std::transform
        std::transform(c_tra_t.begin(), c_tra_t.end(), std::back_inserter(blocks),
                       [](const auto& i) { return ParamBlock{i.data(), i.size(), 3}; });

        return blocks;
    }

    void populate_result(IntrinsicsOptimizationResult<CameraT>& result) const {
        const size_t num_views = c_quat_t.size();
        result.c_se3_t.resize(num_views);

        result.camera = CameraTraits<CameraT>::template from_array<double>(intr.data());
        for (size_t v = 0; v < num_views; ++v) {
            result.c_se3_t[v] = restore_pose(c_quat_t[v], c_tra_t[v]);
        }
    }
};

template <camera_model CameraT>
static ceres::Problem build_problem(const std::vector<PlanarView>& views,
                                    const IntrinsicsOptions& opts,
                                    IntrinsicBlocks<CameraT>& blocks) {
    ceres::Problem p;
    for (size_t view_idx = 0; view_idx < views.size(); ++view_idx) {
        const auto& view = views[view_idx];
        auto* loss = opts.huber_delta > 0 ? new ceres::HuberLoss(opts.huber_delta) : nullptr;
        p.AddResidualBlock(IntrinsicResidual<CameraT>::create(view), loss,
                           blocks.c_quat_t[view_idx].data(), blocks.c_tra_t[view_idx].data(),
                           blocks.intr.data());
    }

    for (auto& c_quat_t : blocks.c_quat_t) {
        p.SetManifold(c_quat_t.data(), new ceres::QuaternionManifold());
    }

    p.SetParameterLowerBound(blocks.intr.data(), CameraTraits<CameraT>::idx_fx, 0.0);
    p.SetParameterLowerBound(blocks.intr.data(), CameraTraits<CameraT>::idx_fy, 0.0);
    if (!opts.optimize_skew) {
        p.SetManifold(blocks.intr.data(),
                      new ceres::SubsetManifold(IntrinsicBlocks<CameraT>::intr_size,
                                                {CameraTraits<CameraT>::idx_skew}));
    }

    return p;
}

static void validate_input(const std::vector<PlanarView>& views) {
    if (views.size() < 4) {
        throw std::invalid_argument("Insufficient views for calibration (at least 4 required).");
    }
}

template <camera_model CameraT>
IntrinsicsOptimizationResult<CameraT> optimize_intrinsics(
    const std::vector<PlanarView>& views, const CameraT& init_camera,
    std::vector<Eigen::Isometry3d> init_c_se3_t, const IntrinsicsOptions& opts) {
    validate_input(views);

    auto blocks = IntrinsicBlocks<CameraT>::create(init_camera, init_c_se3_t);
    ceres::Problem problem = build_problem(views, opts, blocks);

    IntrinsicsOptimizationResult<CameraT> result;
    solve_problem(problem, opts, &result);

    blocks.populate_result(result);
    if (opts.compute_covariance) {
        auto optcov = compute_covariance(blocks, problem);
        if (optcov.has_value()) {
            result.covariance = std::move(optcov.value());
        }
    }

    return result;
}

template IntrinsicsOptimizationResult<PinholeCamera<BrownConradyd>> optimize_intrinsics(
    const std::vector<PlanarView>& views, const PinholeCamera<BrownConradyd>& init_camera,
    std::vector<Eigen::Isometry3d> init_c_se3_t, const IntrinsicsOptions& opts);

template IntrinsicsOptimizationResult<ScheimpflugCamera<PinholeCamera<BrownConradyd>>>
optimize_intrinsics(const std::vector<PlanarView>& views,
                    const ScheimpflugCamera<PinholeCamera<BrownConradyd>>& init_camera,
                    std::vector<Eigen::Isometry3d> init_c_se3_t, const IntrinsicsOptions& opts);

static bool view_has_noncollinear(const PlanarView& view) {
    if (view.size() < 3) {
        return false;
    }
    for (size_t i = 0; i < view.size() - 2; ++i) {
        for (size_t j = i + 1; j < view.size() - 1; ++j) {
            for (size_t k = j + 1; k < view.size(); ++k) {
                const Eigen::Vector2d a = view[j].object_xy - view[i].object_xy;
                const Eigen::Vector2d b = view[k].object_xy - view[i].object_xy;
                if (std::abs(a.x() * b.y() - a.y() * b.x()) > 1e-9) {
                    return true;
                }
            }
        }
    }
    return false;
}

static Eigen::Matrix<double, 6, 1> v_ij(const Eigen::Matrix3d& H, int i, int j) {
    Eigen::Matrix<double, 6, 1> v;
    v << H(0, i) * H(0, j), H(0, i) * H(1, j) + H(1, i) * H(0, j), H(1, i) * H(1, j),
        H(2, i) * H(0, j) + H(0, i) * H(2, j), H(2, i) * H(1, j) + H(1, i) * H(2, j),
        H(2, i) * H(2, j);
    return v;
}

IntrinsicsResult extimate_intrinsics_from_planar(const std::vector<PlanarView>& views,
                                                 const ImageSize& image_size,
                                                 const CalibrateIntrinsicsOptions& opts) {
    if (views.empty()) {
        throw std::invalid_argument("No calibration views provided");
    }

    const double cx0 = static_cast<double>(image_size.width) / 2.0;
    const double cy0 = static_cast<double>(image_size.height) / 2.0;

    std::vector<PlanarView> centered = views;
    for (auto& v : centered) {
        if (v.size() < 6) {
            throw std::invalid_argument("Each view must contain at least 6 points");
        }
        if (!view_has_noncollinear(v)) {
            throw std::invalid_argument("Planar correspondences must be non-collinear");
        }
        if (opts.recenter) {
            for (auto& ob : v) {
                ob.image_uv.x() -= cx0;
                ob.image_uv.y() -= cy0;
            }
        }
    }

    // Step 2: per-view homographies
    std::vector<Eigen::Matrix3d> homographies;
    homographies.reserve(centered.size());
    for (const auto& view : centered) {
        std::vector<Eigen::Vector2d> obj(view.size());
        std::vector<Eigen::Vector2d> img(view.size());
        for (size_t i = 0; i < view.size(); ++i) {
            obj[i] = view[i].object_xy;
            img[i] = view[i].image_uv;
        }
        homographies.push_back(estimate_homography_dlt(obj, img));
    }

    // Step 3: Zhang initialization for K
    Eigen::MatrixXd V(2 * homographies.size(), 6);
    for (size_t i = 0; i < homographies.size(); ++i) {
        const Eigen::Matrix3d& H = homographies[i];
        V.row(2 * i) = v_ij(H, 0, 1).transpose();
        V.row(2 * i + 1) = (v_ij(H, 0, 0) - v_ij(H, 1, 1)).transpose();
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(V, Eigen::ComputeFullV);
    Eigen::VectorXd b = svd.matrixV().col(5);

    double B11 = b[0];
    double B12 = b[1];
    double B22 = b[2];
    double B13 = b[3];
    double B23 = b[4];
    double B33 = b[5];
    double v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 * B12);
    double lambda = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11;
    double alpha = std::sqrt(lambda / B11);
    double beta = std::sqrt(lambda * B11 / (B11 * B22 - B12 * B12));
    double gamma = -B12 * alpha * alpha * beta / lambda;
    double u0 = gamma * v0 / beta - B13 * alpha * alpha / lambda;

    CameraMatrix kmtx{alpha, beta, u0, v0, gamma};
    if (opts.recenter) {
        kmtx.cx += cx0;
        kmtx.cy += cy0;
    }

    // Step 4: Extrinsics per view
    Eigen::Matrix3d K;
    K << kmtx.fx, kmtx.skew, kmtx.cx, 0.0, kmtx.fy, kmtx.cy, 0.0, 0.0, 1.0;
    Eigen::Matrix3d Kinv = K.inverse();
    IntrinsicsResult result;
    for (const auto& H : homographies) {
        Eigen::Matrix3d Hc = H;
        Eigen::Vector3d h1 = Hc.col(0);
        Eigen::Vector3d h2 = Hc.col(1);
        Eigen::Vector3d h3 = Hc.col(2);
        Eigen::Vector3d r1 = Kinv * h1;
        Eigen::Vector3d r2 = Kinv * h2;
        double s = 1.0 / r1.norm();
        r1 *= s;
        r2 *= s;
        Eigen::Vector3d r3 = r1.cross(r2);
        Eigen::Matrix3d R;
        R.col(0) = r1;
        R.col(1) = r2;
        R.col(2) = r3;
        Eigen::JacobiSVD<Eigen::Matrix3d> rsvd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        R = rsvd.matrixU() * rsvd.matrixV().transpose();
        Eigen::Vector3d t = s * (Kinv * h3);
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.linear() = R;
        pose.translation() = t;
        result.c_se3_t.push_back(pose);
    }

    // Step 5: Initial radial distortion (linear LS)
    size_t total_obs = 0;
    for (const auto& v : centered) {
        total_obs += v.size();
    }
    Eigen::MatrixXd A(2 * total_obs, opts.num_radial);
    Eigen::VectorXd bb(2 * total_obs);
    size_t row = 0;
    for (size_t view_idx = 0; view_idx < centered.size(); ++view_idx) {
        const auto& view = centered[view_idx];
        const Eigen::Isometry3d& pose = result.c_se3_t[view_idx];
        for (const auto& ob : view) {
            Eigen::Vector3d Pw(ob.object_xy.x(), ob.object_xy.y(), 0.0);
            Eigen::Vector3d Pc = pose * Pw;
            double x = Pc.x() / Pc.z();
            double y = Pc.y() / Pc.z();
            double u = ob.image_uv.x();
            double v = ob.image_uv.y();
            double dx = (u - kmtx.cx) / kmtx.fx - x;
            double dy = (v - kmtx.cy) / kmtx.fy - y;
            double r2 = x * x + y * y;
            double rpow = r2;
            for (int k = 0; k < opts.num_radial; ++k) {
                A(row, k) = x * rpow;
                A(row + 1, k) = y * rpow;
                rpow *= r2;
            }
            bb(row) = dx;
            bb(row + 1) = dy;
            row += 2;
        }
    }
    Eigen::VectorXd dist = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(bb);
    Eigen::VectorXd coeffs = Eigen::VectorXd::Zero(opts.num_radial + 2);
    coeffs.head(opts.num_radial) = dist.head(opts.num_radial);

    result.camera = PinholeCamera<BrownConradyd>(kmtx, coeffs);
    return result;
}

}  // namespace calib
