# NPZ contract versioning

Two file contracts separate the packages and the upper-computer consumer.

## Source NPZ

Contract ID: `external_layer_paths_v1`

The `kuka-slicer` writer records this ID in `meta.format`. The
`external_npz_preprocessor` reader accepts that ID and continues to accept
historical source files without `meta.format`. An explicit unknown ID is
rejected instead of being silently interpreted as v1.

This is the only integration boundary between `kuka-slicer` and
`kuka-offline-planner`; neither package imports the other.

## System NPZ

Contract ID in code: `kuka_system_trajectory_v1`

The pre-existing system NPZ format has no metadata field in the archive.
Version 1 is therefore detected from its frozen set of 25 fields, exact dtypes,
one-dimensional shapes, common row count, and the two vocabularies. The
validator is implemented in `path_processing_core.npz_contract`.

No field was added to the archive during versioning. Existing golden files and
the current upper-computer loader are therefore version-1 files without a
migration. Additional unknown fields are tolerated so a future additive
extension does not break a v1 reader; changing or removing a v1 field requires
a new contract version and separate consumer approval.
