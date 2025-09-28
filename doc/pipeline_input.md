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

These artifacts enable downstream tools to consume calibrated intrinsics and
extrinsics without re-running the pipeline.

