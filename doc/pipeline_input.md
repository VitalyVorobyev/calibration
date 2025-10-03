# Planar Intrinsics + Stereo Extrinsics Pipeline Input

This document describes the JSON format consumed by the intrinsic/extrinsic
calibration pipeline example (`calib_example_intrinsic_extrinsic`).  The format
combines planar intrinsics data with stereo pair definitions so the pipeline
can recover per-camera intrinsics and relative poses in a single run.

## Top-Level Structure

```json
{
  "planar_intrinsics_config": "planar_intrinsics_config.json",
  "planar_detections": [
    { "sensor_id": "cam0", "path": "features/cam0.json" },
    { "sensor_id": "cam1", "path": "features/cam1.json" }
  ],
  "stereo": {
    "pairs": [
      {
        "pair_id": "rig",
        "reference_sensor": "cam0",
        "target_sensor": "cam1",
        "views": [
          { "reference_image": "ref_view0.json", "target_image": "tgt_view0.json" }
        ],
        "options": {
          "optimize_intrinsics": false,
          "optimize_skew": false,
          "optimize_extrinsics": true,
          "huber_delta": 1.0,
          "max_iterations": 100,
          "compute_covariance": true,
          "verbose": false
        }
      }
    ]
  }
}
```

### `planar_intrinsics_config`

Path (relative to the JSON file or absolute) to the planar intrinsics
configuration consumed by `planar::load_calibration_config`.  This file
describes per-camera calibration options such as minimum corner counts, RANSAC
settings and distortion bounds.

### `planar_detections`

An array of feature-detection files. Each element must contain:

* `sensor_id` – camera identifier used in the intrinsics config and stereo
  definitions.
* `path` – JSON file containing planar feature detections (same format as the
  existing planar intrinsics example).

The pipeline assigns detections to cameras via this `sensor_id`. All stereo
views must reference `PlanarImageDetections.file` values contained in these
files.

### `stereo.pairs`

Defines every stereo rig to calibrate. Each pair entry includes:

* `pair_id` – unique identifier used when storing artifacts.
* `reference_sensor` – camera treated as the reference (identity pose).
* `target_sensor` – second camera in the pair.
* `views` – list of matched captures. Every view maps a reference image file to
  a target image file. The filenames must match the `file` field inside the
  corresponding detection JSON.
* `options` – optional solver parameters (mapped to `ExtrinsicOptions`). All
  fields are optional and fall back to library defaults.

The pipeline filters each view to ensure both cameras provide at least four
planar correspondences. Views with insufficient data are reported but ignored
during optimisation.

### `hand_eye`

Optional configuration enabling the hand-eye calibration stage in the pipeline
examples. The section accepts either a single rig definition or an object with
`rigs`, mirroring the structures in `include/calib/pipeline/handeye.h`:

```json
"hand_eye": {
  "rig_id": "arm",
  "sensors": ["cam0"],
  "min_angle_deg": 1.0,
  "options": {
    "max_iterations": 80,
    "huber_delta": 1.0,
    "compute_covariance": true
  },
  "observations": [
    {
      "id": "pose0",
      "base_se3_gripper": [[1,0,0,0],[0,1,0,0],[0,0,1,0.5],[0,0,0,1]],
      "images": { "cam0": "cam0_pose0.json" }
    }
  ]
}
```

Each observation links a robot pose (`base_se3_gripper`) with the image files
containing planar detections for the participating sensors. The filenames must
match the `file` entries stored in the planar detection datasets.

### `bundle`

Enables the bundle-adjustment refinement stage. The format mirrors the hand-eye
configuration and reuses observations when the section omits them:

```json
"bundle": {
  "rig_id": "arm",
  "sensors": ["cam0", "cam1"],
  "options": {
    "optimize_intrinsics": false,
    "optimize_target_pose": true,
    "optimize_hand_eye": true,
    "max_iterations": 80
  }
}
```

If `observations` is absent the pipeline falls back to the hand-eye rig with
the same identifier, avoiding duplication of the robot pose metadata.

## Generated Artifacts

The example application writes the full artifact bundle to the `--output`
location. The JSON contains:

* `pipeline_summary` – execution status for each stage.
* `stereo.pairs.<pair_id>.optimization` – serialised
  `ExtrinsicOptimizationResult`, including refined camera parameters and
  relative poses.
* `stereo.pairs.<pair_id>.initial_guess` – initial poses estimated by DLT.
* `stereo.pairs.<pair_id>.views` – per-view bookkeeping (point counts and
  gating status).
* `hand_eye.<rig_id>.result` – optimised hand-eye transform(s) and solver
  diagnostics when the stage is enabled.
* `bundle.<rig_id>.result` – bundle-adjustment output, including refined target
  pose and per-sensor hand-eye transforms.

These artifacts enable downstream tools to consume calibrated intrinsics and
extrinsics without re-running the pipeline.
