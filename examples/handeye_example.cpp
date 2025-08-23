#include <iostream>
#include <vector>

#include "calibration/handeye.h"

using namespace vitavision;

int main() {
    HandEyeOptions opts;
    opts.optimize_extrinsics = true;
    std::vector<Eigen::Affine3d> initial_ext; // empty for single camera
    HandEyeResult res = calibrate_hand_eye(initial_ext, opts);
    std::cout << res.summary << "\n";
    return 0;
}

