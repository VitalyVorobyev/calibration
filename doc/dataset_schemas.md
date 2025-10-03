# Calibration Dataset Schema

The calibration pipeline consumes feature datasets encoded as JSON.  For planar
targets the format is intentionally lightweight: the structures exposed by
`calib::planar` are simple aggregates and map directly to JSON via
*nlohmann::json* helpers.

## Planar Dataset Overview

```json
{
  "image_directory": "…",
  "feature_type": "planar",
  "algo_version": "4.8.0",
  "params_hash": "0123abcd4567ef89",
  "sensor_id": "cam0",
  "tags": ["recorded"],
  "metadata": {
    "detector": {
      "name": "opencv.findChessboardCorners"
    }
  },
  "images": [
    {
      "file": "img_0001.png",
      "points": [
        {"id": 0, "x": 512.1, "y": 412.7, "local_x": 0.0, "local_y": 0.0, "local_z": 0.0},
        {"id": 1, "x": 642.5, "y": 410.3, "local_x": 0.025, "local_y": 0.0, "local_z": 0.0}
      ]
    }
  ]
}
```

### Key Concepts

- **image_directory** – optional helper that points to the folder containing
  the raw images for convenience in reporting.
- **feature_type**, **algo_version**, **params_hash** – metadata emitted by the
  detector.  Empty strings are acceptable when the information is unknown.
- **sensor_id** – the logical camera identifier associated with every image in
  the dataset.
- **tags** – arbitrary labels such as `synthetic` or `recorded` that the
  pipeline uses for gating decisions.
- **images** – flat list of detections.  Each entry binds a file name to
  resolved correspondences expressed as planar target points.

## Runtime Validation

`calib::PlanarDetections` integrates with *nlohmann::json* via
`from_json`/`to_json` specialisations.  The conversion simply maps struct
members to JSON fields, relying on nlohmann's built-in error handling—missing or
type-mismatched fields raise `nlohmann::json::exception` automatically.

## Recommended Workflow

1. Export detector results directly using the structure outlined above.
2. Optionally validate the JSON by running a round-trip through the
   `PlanarDetections` JSON converters.
3. Feed the dataset into the new calibration pipeline via
   `calib::pipeline::JsonPlanarDatasetLoader`.

The schema is intentionally minimal and can be extended gradually—additional
fields may be embedded under `metadata` or attached to individual image
detections without breaking existing tooling.
