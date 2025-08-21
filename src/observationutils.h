/** @brief Utility functions for working with image observations */

#pragma once

// ceres
#include "ceres/rotation.h"

#include "calibration/planarpose.h"

namespace vitavision {

template<typename T>
Observation<T> to_observation(const PlanarObservation& obs, const T* pose6) {
    const T* aa = pose6;      // angle-axis
    const T* t  = pose6 + 3;  // translation

    Eigen::Matrix<T, 3, 1> P {T(obs.object_xy.x()), T(obs.object_xy.y()), T(0.0)};
    Eigen::Matrix<T, 3, 1> Pc;
    ceres::AngleAxisRotatePoint(aa, P.data(), Pc.data());
    Pc += Eigen::Matrix<T, 3, 1>(t[0], t[1], t[2]);
    T invZ = T(1.0) / Pc.z();
    Observation<T> ob;
    ob.x = Pc.x() * invZ;
    ob.y = Pc.y() * invZ;
    ob.u = T(obs.image_uv.x());
    ob.v = T(obs.image_uv.y());
    return ob;
}

}  // namespace vitavision
