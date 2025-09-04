#pragma once

#ifndef CALIB_EXPORT
#define CALIB_EXPORT
#endif

#include <Eigen/Core>
#include <memory>
#include <typeinfo>

#include "calib/cameramodel.h"

namespace calib {

class CALIB_EXPORT AnyCamera {
public:
  AnyCamera() = default;

  template <CameraModel M>
  AnyCamera(M cam)
      : self_(std::make_shared<Model<M>>(std::move(cam))),
        type_(&typeid(M)) {}

  bool has_value() const noexcept { return static_cast<bool>(self_); }

  Eigen::Vector2d project(const Eigen::Vector3d& Xc) const {
    return self_->project(Xc);
  }
  Eigen::Vector3d unproject(const Eigen::Vector2d& xn) const {
    return self_->unproject(xn);
  }

  template <class M>
  bool holds() const noexcept {
    return type_ && *type_ == typeid(M);
  }

  template <class M>
  M* as() {
    if (!holds<M>()) return nullptr;
    // Safe because we just checked the exact stored type
    return &static_cast<Model<M>*>(self_.get())->cam;
  }

private:
  struct Concept {
    virtual ~Concept() = default;
    virtual Eigen::Vector2d project(const Eigen::Vector3d&) const = 0;
    virtual Eigen::Vector3d unproject(const Eigen::Vector2d&) const = 0;
  };

  template <CameraModel M>
  struct Model final : Concept {
    explicit Model(M c) : cam(std::move(c)) {}
    Eigen::Vector2d project(const Eigen::Vector3d& X) const override {
      return cam.project(X);
    }
    Eigen::Vector3d unproject(const Eigen::Vector2d& x) const override {
      return cam.unproject(x);
    }
    M cam;
  };

  std::shared_ptr<Concept> self_;
  const std::type_info* type_{nullptr};
};

} // namespace calib

