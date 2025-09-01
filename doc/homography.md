# Homography Estimation

The `homography` component estimates planar homographies and refines them for
pose recovery and mapping tasks.

## Algorithms

* **Direct Linear Transform** – `estimate_homography_dlt` solves for the
  homography matrix using the classic DLT algorithm with point correspondences.
* **Non-linear Refinement** – `optimize_homography` performs least-squares
  optimisation with optional robust loss and covariance estimation.

## API

```
Eigen::Matrix3d estimate_homography_dlt(
    const std::vector<Eigen::Vector2d>& src,
    const std::vector<Eigen::Vector2d>& dst);

OptimizeHomographyResult optimize_homography(
    const std::vector<Eigen::Vector2d>& src,
    const std::vector<Eigen::Vector2d>& dst,
    const Eigen::Matrix3d& init,
    const HomographyOptions& opts = {});
```
