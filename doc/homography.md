# Homography Estimation

The `homography` component estimates planar homographies and refines them for
pose recovery and mapping tasks.

## Algorithms

* **Direct Linear Transform** – `estimate_homography_dlt` solves for the
  homography matrix using the classic DLT algorithm with point correspondences.
* **Non-linear Refinement** – `optimize_homography` performs least-squares
  optimisation to improve accuracy and robustness to noise.

## API

```
Eigen::Matrix3d estimate_homography_dlt(
    const std::vector<Eigen::Vector2d>& src,
    const std::vector<Eigen::Vector2d>& dst);

Eigen::Matrix3d optimize_homography(
    const std::vector<Eigen::Vector2d>& src,
    const std::vector<Eigen::Vector2d>& dst);
```
