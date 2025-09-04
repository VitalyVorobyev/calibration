#pragma once

#ifndef CALIB_EXPORT
#define CALIB_EXPORT
#endif

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <memory>
#include <typeinfo>
#include <variant>

#include "calib/camera.h"
#include "calib/scheimpflug.h"
#include "calib/distortion.h"
#include "calib/cameramodel.h"

namespace calib {

struct CameraModelTraits final {
  int param_count{0};
  int idx_fx{0};
  int idx_fy{0};
  int idx_skew{0};
};

// Type-erased camera wrapper with runtime parameters
class CALIB_EXPORT AnyCamera {
 public:
  AnyCamera() = default;

  template <CameraModel M>
  AnyCamera(M cam) : cam_(std::move(cam)) {
    traits_ = {static_cast<int>(CameraTraits<M>::param_count),
               CameraTraits<M>::idx_fx,
               CameraTraits<M>::idx_fy,
               CameraTraits<M>::idx_skew};
    params_.resize(traits_.param_count);
    std::array<double, CameraTraits<M>::param_count> arr{};
    CameraTraits<M>::to_array(std::get<M>(cam_), arr);
    for (int i = 0; i < traits_.param_count; ++i) {
      params_[static_cast<Eigen::Index>(i)] = arr[static_cast<size_t>(i)];
    }
  }

  bool has_value() const noexcept {
    return !std::holds_alternative<std::monostate>(cam_);
  }

  template <class M>
  bool holds() const noexcept {
    return std::holds_alternative<M>(cam_);
  }

  template <class M>
  M* as() {
    return std::get_if<M>(&cam_);
  }

  const CameraModelTraits& traits() const { return traits_; }
  Eigen::VectorXd& params() { return params_; }
  const Eigen::VectorXd& params() const { return params_; }

  template <typename T>
  Eigen::Matrix<T, 2, 1> project(const Eigen::Matrix<T, 3, 1>& X,
                                  const T* intr) const {
    return std::visit(
        [&](const auto& cam) -> Eigen::Matrix<T, 2, 1> {
          using CamT = std::decay_t<decltype(cam)>;
          if constexpr (std::is_same_v<CamT, std::monostate>) {
            return Eigen::Matrix<T, 2, 1>::Zero();
          } else {
            auto c = CameraTraits<CamT>::template from_array<T>(intr);
            return c.project(X);
          }
        },
        cam_);
  }

  template <typename T>
  Eigen::Matrix<T, 3, 1> unproject(const Eigen::Matrix<T, 2, 1>& x,
                                    const T* intr) const {
    return std::visit(
        [&](const auto& cam) -> Eigen::Matrix<T, 3, 1> {
          using CamT = std::decay_t<decltype(cam)>;
          if constexpr (std::is_same_v<CamT, std::monostate>) {
            return Eigen::Matrix<T, 3, 1>::Zero();
          } else {
            auto c = CameraTraits<CamT>::template from_array<T>(intr);
            Eigen::Matrix<T, 3, 1> out;
            auto xy = c.unproject(x);
            out << xy.x(), xy.y(), T(1);
            return out;
          }
        },
        cam_);
  }

  Eigen::Vector2d project(const Eigen::Vector3d& X) const {
    return project<double>(X, params_.data());
  }

  Eigen::Vector3d unproject(const Eigen::Vector2d& x) const {
    return unproject<double>(x, params_.data());
  }

 private:
  using CameraVariant =
      std::variant<std::monostate, Camera<BrownConradyd>,
                   ScheimpflugCamera<BrownConradyd>, Camera<DualDistortion>>;

  CameraVariant cam_;
  CameraModelTraits traits_;
  Eigen::VectorXd params_;
};

}  // namespace calib
