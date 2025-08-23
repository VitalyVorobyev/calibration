# Camera Model

The `camera` component combines a `CameraMatrix` with distortion coefficients to
form a simple pinhole camera model. It provides utilities for projecting points
using the intrinsic and distortion parameters.

## Algorithms

* **Distortion Application** – Normalized coordinates are warped using
  `apply_distortion` to account for radial and tangential distortion.
* **Denormalisation** – Distorted normalized coordinates are converted to pixel
  coordinates through the intrinsic matrix.

## API

```
struct Camera {
    CameraMatrix intrinsics;
    Eigen::VectorXd distortion;

    template <typename T>
    Eigen::Matrix<T,2,1> project_normalized(
        const Eigen::Matrix<T,2,1>& xyn) const;
};
```

`project_normalized` projects a point in normalized image coordinates to pixel
coordinates by applying distortion and then denormalising using the camera
matrix.
