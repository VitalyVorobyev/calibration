#include "calib/pipeline/linescan.h"

namespace calib::pipeline {

static PinholeCamera<DualDistortion> to_dual(const PinholeCamera<BrownConradyd>& cam) {
    DualDistortion dual;
    dual.forward = cam.distortion.coeffs;
    dual.inverse = invert_brown_conrady(cam.distortion.coeffs);
    return {cam.kmtx, dual};
}

auto LinescanCalibrationFacade::calibrate(const PinholeCamera<BrownConradyd>& camera,
                                          const std::vector<LineScanView>& views,
                                          const LinescanCalibrationOptions& opts) const
    -> LinescanCalibrationRunResult {
    LinescanCalibrationRunResult out;
    out.used_views = views.size();
    try {
        out.result = calibrate_laser_plane(views, to_dual(camera), opts.plane_fit);
        out.success = true;
    } catch (...) {
        out.success = false;
    }
    return out;
}

}  // namespace calib::pipeline
