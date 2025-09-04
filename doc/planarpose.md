# Planar Pose Estimation

The `planarpose` component estimates the pose of a planar target relative to the
camera from pixel observations.

## Algorithms

* **Homography Decomposition** – `pose_from_homography_normalized` converts a
  homography in normalized coordinates into a 3D pose.
* **Direct DLT Pose** – `estimate_planar_pose_dlt` forms a homography from
  object and image points using the camera matrix and decomposes it to a pose.
* **Non-linear Refinement** – `optimize_planar_pose` jointly estimates target
  pose and lens distortion using least-squares with covariance estimation.

## API

```
Eigen::Isometry3d pose_from_homography_normalized(const Eigen::Matrix3d& H);

Eigen::Isometry3d estimate_planar_pose_dlt(
    const std::vector<Eigen::Vector2d>& obj_xy,
    const std::vector<Eigen::Vector2d>& img_uv,
    const CameraMatrix& intrinsics);

PlanarPoseResult optimize_planar_pose(
    const std::vector<Eigen::Vector2d>& obj_xy,
    const std::vector<Eigen::Vector2d>& img_uv,
    const CameraMatrix& intrinsics,
    const PlanarPoseOptions& opts = {});
```
