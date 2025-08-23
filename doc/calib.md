# Camera Calibration

The `calib` component performs full single camera calibration from multiple
planar views. It exposes the **`calibrate_camera_planar`** function which takes a
collection of `PlanarView` observations and estimates camera intrinsics,
distortion coefficients and the pose of each view.

## Algorithms

1. **Homography and Pose Estimation** – Each planar view is converted to a
   homography and decomposed to obtain an initial pose estimate.
2. **Non-linear Optimization** – Intrinsics, distortion and poses are refined
   jointly using bundle adjustment to minimise reprojection error.
3. **Covariance Analysis** – After optimisation the covariance matrix of all
   estimated parameters is computed to provide uncertainty information.

## API

```
CameraCalibrationResult calibrate_camera_planar(
    const std::vector<PlanarView>& views,
    int num_radial,
    const CameraMatrix& initial_guess,
    bool verbose = false,
    std::optional<CalibrationBounds> bounds = std::nullopt);
```

The returned `CameraCalibrationResult` contains the optimised camera matrix,
distortion coefficients, per-view poses, covariance and error statistics.
