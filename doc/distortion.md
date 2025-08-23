# Lens Distortion

The `distortion` component provides functions for applying and estimating lens
 distortion parameters.

## Algorithms

* **Distortion Model** – Supports an arbitrary number of radial coefficients plus
tangential terms `p1` and `p2`. Distortion is applied to normalized coordinates
with `apply_distortion`.
* **Linear Least Squares Fit** – `fit_distortion_full` constructs a design
matrix from observations and solves for distortion coefficients using SVD.
* **Residual Analysis** – The solver returns residuals for diagnostic purposes.

## API

```
template<typename T>
Eigen::Matrix<T,2,1> apply_distortion(
    const Eigen::Matrix<T,2,1>& norm_xy,
    const Eigen::VectorXd& coeffs);

std::optional<DistortionWithResiduals<T>> fit_distortion_full(
    const std::vector<Observation<T>>& obs,
    T fx, T fy, T cx, T cy,
    int num_radial = 2);
```

These utilities model and estimate distortion to support downstream calibration
steps.
