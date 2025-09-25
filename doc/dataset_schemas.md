# Calibration Dataset Schema

The calibration pipeline consumes feature datasets encoded as JSON.  The base
specification is versioned and designed to cover single- and multi-sensor
setups.  The authoritative schema is stored at
`schemas/calib_dataset.schema.json` and follows JSON Schema draft 2020-12.

## Version 1 Overview

```json
{
  "schema_version": 1,
  "feature_type": "planar_points",
  "metadata": {
    "image_directory": "…",
    "detector": {
      "type": "planar",
      "name": "opencv.findChessboardCorners",
      "version": "4.8.0",
      "params_hash": "0123abcd4567ef89"
    }
  },
  "targets": [
    { "id": "board", "type": "planar", "rows": 7, "cols": 10, "spacing": 0.025 }
  ],
  "captures": [
    {
      "sensor_id": "cam0",
      "frame": "img_0001.png",
      "timestamp": 1712570421.34,
      "tags": ["recorded"],
      "observations": [
        {
          "target_id": "board",
          "type": "planar_points",
          "points": [
            {"id": 0, "pixel": [512.1, 412.7], "target": [0.0, 0.0, 0.0]},
            {"id": 1, "pixel": [642.5, 410.3], "target": [0.025, 0.0, 0.0]}
          ]
        }
      ]
    }
  ]
}
```

### Key Concepts

- **schema_version** – allows backwards compatible evolution.  Version 1 focuses
  on planar correspondences.
- **feature_type** – indicates the observation modality.  Future revisions can
  introduce additional types while reusing the same capture structure.
- **captures** – minimal grouping of feature measurements.  Each capture binds a
  frame identifier to a `sensor_id` and carries a list of observations.  Tags
  label captures (for example `synthetic` vs `recorded`) enabling pipeline
  gating.
- **observations** – a typed payload.  `planar_points` encodes pixel ↔ board
  correspondences where `target` holds `[X, Y]` or `[X, Y, Z]` coordinates.

## Legacy Conversion

Earlier versions of the library expected detector output with root fields such
as `image_directory`, `feature_type`, `algo_version`, and a flat `images`
array.  Use `calib::planar::convert_legacy_planar_features` to transform those
files into the version 1 dataset.  The helper fills in detector metadata and
marks captures with a `recorded` tag.  Empty frames are skipped to satisfy the
new schema requirements.

## Runtime Validation

`calib::planar::validate_planar_dataset` performs lightweight structural checks
before the data is consumed.  `load_planar_dataset` automatically invokes
validation and reports human-readable error messages when the dataset is
malformed.  The helper understands both the new schema and legacy input via the
conversion utility.

## Recommended Workflow

1. Export detector results to the version 1 schema (or run the conversion
   helper on the legacy format).
2. Optionally validate the JSON using either the shipped schema or the runtime
   validator.
3. Feed the dataset into the new calibration pipeline via
   `calib::pipeline::JsonPlanarDatasetLoader`.

The schema is intentionally minimal and can be extended gradually—additional
fields may be embedded under `metadata`, `capture.metadata`, or within new
observation types without breaking existing tooling.
