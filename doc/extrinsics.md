# Extrinsic Calibration

The `extrinsics` component estimates camera and target poses using planar
observations from multiple cameras.

## Algorithms

1. **Initial Guess** – `make_initial_extrinsic_guess` triangulates planar poses
   to seed camera and target transforms.
2. **Joint Optimisation** – `optimize_joint_intrinsics_extrinsics` refines
   intrinsics, distortion and all poses simultaneously to minimise pixel error.
3. **Pose-Only Refinement** – `optimize_extrinsic_poses` adjusts camera and
   target poses while keeping intrinsics fixed.
4. **Covariance Computation** – Covariances for each pose and intrinsics are
   returned for uncertainty analysis.

## API

```
InitialExtrinsicGuess make_initial_extrinsic_guess(
    const std::vector<ExtrinsicPlanarView>& views,
    const std::vector<Camera>& cameras);

JointOptimizationResult optimize_joint_intrinsics_extrinsics(
    const std::vector<ExtrinsicPlanarView>& views,
    const std::vector<Camera>& initial_cameras,
    const std::vector<Eigen::Affine3d>& initial_camera_poses,
    const std::vector<Eigen::Affine3d>& initial_target_poses,
    bool verbose = false);

ExtrinsicOptimizationResult optimize_extrinsic_poses(
    const std::vector<ExtrinsicPlanarView>& views,
    const std::vector<Camera>& cameras,
    const std::vector<Eigen::Affine3d>& initial_camera_poses,
    const std::vector<Eigen::Affine3d>& initial_target_poses,
    bool verbose = false);
```
