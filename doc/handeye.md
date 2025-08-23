# Hand-Eye Calibration

The `handeye` component estimates the rigid transform between a robot gripper
and a camera using planar target observations.

## Algorithms

1. **Tsai–Lenz Initialisation** – `estimate_hand_eye_initial` provides a closed
   form solution from corresponding gripper and target poses.
2. **Bundle Adjustment** – `calibrate_hand_eye` refines intrinsics, the
   gripper-to-camera transform, optional target pose and extrinsics through a
   non-linear optimisation.
3. **Covariance Estimation** – The final solver computes covariance matrices for
   pose parameters enabling accuracy assessment.

## API

```
Eigen::Affine3d estimate_hand_eye_initial(
    const std::vector<Eigen::Affine3d>& base_T_gripper,
    const std::vector<Eigen::Affine3d>& target_T_camera);

HandEyeResult calibrate_hand_eye(
    const std::vector<HandEyeObservation>& observations,
    const std::vector<CameraMatrix>& initial_intrinsics,
    const Eigen::Affine3d& initial_hand_eye,
    const std::vector<Eigen::Affine3d>& initial_extrinsics = {},
    const Eigen::Affine3d& initial_base_target = Eigen::Affine3d::Identity(),
    const HandEyeOptions& opts = {});
```
