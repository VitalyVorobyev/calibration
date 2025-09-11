# Intrinsic Calibration

The `intrinsics` component focuses on estimating camera matrix parameters and
lens distortion from point observations.

## Algorithms

1. **Planar View Initialisation** – `estimate_intrinsics` computes an initial
   camera matrix and per-view poses from planar observations by combining
   homography decomposition with the linear solver.
2. **Linear Estimation** – `estimate_intrinsics_linear` solves a least-squares
   system for the camera matrix ignoring distortion, optionally with parameter
   bounds.
3. **Iterative Refinement** – `estimate_intrinsics_linear_iterative` alternates
   between solving for distortion (`fit_distortion`) and recomputing intrinsics
   for a robust initialisation.
4. **Non-linear Optimisation** – `optimize_intrinsics` refines intrinsics and
   distortion simultaneously while computing covariance and reprojection error
   statistics.

## API

```
std::optional<IntrinsicsEstimateResult> estimate_intrinsics(
    const std::vector<PlanarView>& views,
    const Eigen::Vector2i& image_size,
    const IntrinsicsEstimateOptions& opts = {});

std::optional<CameraMatrix> estimate_intrinsics_linear(
    const std::vector<Observation<double>>& obs,
    std::optional<CalibrationBounds> bounds = std::nullopt);

std::optional<LinearInitResult> estimate_intrinsics_linear_iterative(
    const std::vector<Observation<double>>& obs,
    int num_radial,
    int max_iterations = 5);

IntrinsicOptimizationResult optimize_intrinsics(
    const std::vector<Observation<double>>& obs,
    int num_radial,
    const CameraMatrix& initial_guess,
    bool verb = false,
    std::optional<CalibrationBounds> bounds = std::nullopt);
```
