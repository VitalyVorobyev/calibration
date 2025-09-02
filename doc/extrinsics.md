# Extrinsic Calibration

The `extrinsics` component estimates camera and target poses using planar
observations from multiple cameras.

## Algorithms

1. **Initial Guess** – `estimate_extrinsic_dlt` triangulates planar poses
   to seed camera and target transforms.
2. **Joint Optimisation** – `optimize_joint_intrinsics_extrinsics` refines
   intrinsics, distortion and all poses simultaneously to minimise pixel error.
3. **Pose-Only Refinement** – `optimize_extrinsic_poses` adjusts camera and
   target poses while keeping intrinsics fixed.
4. **Covariance Computation** – Covariances for each pose and intrinsics are
   returned for uncertainty analysis.

## API

```
InitialExtrinsicGuess estimate_extrinsic_dlt(
    const std::vector<MulticamPlanarView>& views,
    const std::vector<Camera>& cameras);

ExtrinsicOptimizationResult optimize_joint_intrinsics_extrinsics(
    const std::vector<MulticamPlanarView>& views,
    const std::vector<Camera>& initial_cameras,
    const std::vector<Eigen::Isometry3d>& initial_camera_poses,
    const std::vector<Eigen::Isometry3d>& initial_target_poses,
    bool verbose = false);

ExtrinsicOptimizationResult optimize_extrinsic_poses(
    const std::vector<MulticamPlanarView>& views,
    const std::vector<Camera>& cameras,
    const std::vector<Eigen::Isometry3d>& initial_camera_poses,
    const std::vector<Eigen::Isometry3d>& initial_target_poses,
    bool verbose = false);
```
