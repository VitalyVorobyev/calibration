# Intrinsic Calibration

The `intrinsics` component focuses on estimating camera matrix parameters and
lens distortion from point observations.

## Algorithms

1. **Linear Estimation** – `estimate_intrinsics_linear` solves a least-squares
   system for the camera matrix ignoring distortion, optionally with parameter
   bounds.
2. **Iterative Refinement** – `estimate_intrinsics_linear_iterative` alternates
   between solving for distortion (`fit_distortion`) and recomputing intrinsics
   for a robust initialisation.
3. **Non-linear Optimisation** – `optimize_intrinsics` refines intrinsics and
   distortion simultaneously while computing covariance and reprojection error
   statistics.

## API

```
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
