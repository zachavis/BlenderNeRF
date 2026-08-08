# Camera Array (CArr) Synced Multiview Export Design

Date: 2026-08-08
Status: Approved for implementation planning

## Summary

Add Camera Array (CArr) as a fourth, independent BlenderNeRF method. CArr renders a synchronized dynamic scene from a fixed array of `N` inward-facing training cameras and from one automatically generated moving test camera. It exports a NeuS-derived per-camera directory layout matching the existing `docs/model` reference artifact, except that v1 writes RGBA images and intentionally omits the reference artifact's mask and depth folders.

CArr remains separate from Camera on Sphere (COS). It may reuse narrow, stable helpers and the proven modal-rendering pattern, but COS does not gain an array mode and its behavior and output remain unchanged.

## Goals

- Render every enabled training camera at every scene animation frame.
- Render one geometry-derived test pose at every scene animation frame.
- Preserve synchronization through the same zero-based image index in every camera directory.
- Support Circle, local +Z Hemisphere, and Sphere mount geometries.
- Uniformly space circle cameras and near-uniformly distribute surface cameras.
- Match the reference NPZ keys, shapes, dtypes, ordering semantics, and fixed-versus-moving pose behavior.
- Provide an inspectable Blender preview using managed camera objects.
- Use a cancellable modal render queue and restore all temporarily changed scene state.
- Leave the finished dataset as a normal directory rather than creating a ZIP archive.

## Non-goals for v1

- Validation or fixed-camera splits.
- Mask or depth output.
- LLFF, D-NeRF JSON, or selectable output formats.
- Video encoding.
- Per-camera intrinsics, lenses, or resolutions.
- User-authored or animated test-camera paths.
- Animated training rigs.
- Arbitrary mesh or curve mount geometry.
- Ellipsoidal scaling, outward-facing cameras, or manual editing of generated cameras.
- Automatic dataset normalization; `scale_mat_i` is identity to match the reference artifact.

## Reference Convention

The ignored `docs/model` artifact is the compatibility contract. Each camera directory contains a numbered image sequence and one `cameras_sphere.npz`. The inspected reference contains 100 `world_mat_i` arrays and 100 `scale_mat_i` arrays per camera:

- `world_mat_i`: shape `(3, 4)`, dtype `float64`.
- `scale_mat_i`: shape `(4, 4)`, dtype `float32`, identity.
- Training-camera world matrices are constant across all indices.
- Test-camera world matrices vary across indices.
- Image names use zero-based indices with a minimum width of three digits, such as `000.png` through `099.png`.

The reference PNG files are RGB. CArr intentionally writes RGBA PNG files because BlenderNeRF's current rendering workflow preserves alpha and the approved design retains it. The NPZ convention remains unchanged.

The format is derived from the [NeuS data convention](https://github.com/Totoro97/NeuS). A future format selector may add other conventions, but it is outside this iteration.

## Architecture

### CArr registration and UI

CArr has its own UI panel and operator, registered alongside SOF, TTC, and COS. Proposed focused modules are:

- `carr_ui.py`: presents CArr controls and starts the operator.
- `carr_geometry.py`: contains deterministic train layouts, test trajectories, and inward-looking orientation math without file or operator lifecycle responsibilities.
- `carr_neus.py`: computes full intrinsics and reference-compatible projection matrices, constructs typed NPZ entries, and writes `cameras_sphere.npz`.
- `carr_operator.py`: validates the request, snapshots state, prepares metadata and render tasks, runs the modal queue, and restores state.

The implementation should extract shared code from COS only when the boundary is narrow and behavior-preserving. It must not turn COS into a multi-mode conditional operator or undertake unrelated refactoring.

### Rig layout

Rig layout code has one responsibility: map a configuration and index to a deterministic local camera position. It does not create Blender objects or write files. This makes the placement algorithms independently testable.

### Preview rig lifecycle

Preview lifecycle code owns a dedicated Blender collection and its generated objects. It creates, updates, rebuilds, and removes only CArr-managed objects. Generated camera transforms are derived from CArr properties; manual edits are not durable.

### NeuS writer

The NeuS writer converts camera transforms and render intrinsics to reference-compatible projection matrices and writes NPZ archives. It does not render images or mutate scene state.

### Modal render queue

The CArr operator prepares explicit render tasks and processes one image per Blender modal timer tick. It uses the same visible-progress and between-image cancellation model proven by COS.

## User Interface and Properties

The `Camera Array CArr` panel exposes:

- `Geometry`: `Circle` by default, `Hemisphere`, or `Sphere`.
- `Camera Count`: integer, default 10, minimum 2.
- `Location`: rig center in world space.
- `Rotation`: orientation applied to the local mount geometry.
- `Radius`: positive uniform distance from every generated camera to the rig center.
- `Lens`: one focal length shared by all train and test cameras.
- `Show Rig`: toggles the managed preview collection.
- `Name`: dataset directory name.
- `PLAY CArr`: begins export.

CArr honors these existing shared controls:

- `Train`: include or omit all `cam_train_*` directories.
- `Test`: include or omit `cam_test`.
- `Render Frames`: render PNG files when enabled; otherwise create metadata and empty `rgb` directories.
- `Save Log File`: optionally write the usual reproducibility log at dataset root, extended with CArr settings.
- `Save Path`: parent directory for the dataset.

At least one of Train or Test must be enabled. The shared AABB, Gaussian Points, File Format, and Path Format controls do not affect CArr's NeuS-only export. Documentation must state this explicitly.

Suggested managed names are:

- Collection: `BlenderNeRF CArr`.
- Mount object: `BlenderNeRF CArr Rig`.
- Training cameras: `BlenderNeRF CArr Train 000`, `001`, and so on.
- Test camera: `BlenderNeRF CArr Test`.

Changing geometry, count, location, rotation, radius, or lens updates or rebuilds the managed preview. If a generated object is missing or invalid, preview update and export rebuild it. The generated rig remains available after successful, failed, or cancelled export. Disabling `Show Rig` removes only CArr-managed preview objects and their unused camera data.

## Geometry

Let `N` be the training-camera count, `k` be a zero-based camera index, `r` be the radius, and `g = pi * (3 - sqrt(5))` be the golden angle. All formulas first produce a unit position in rig-local space. The final world position is:

`world_position = rig_location + rig_rotation @ (r * local_position)`

### Circle

The circle lies in the rig-local XY plane:

- `theta = 2*pi*k/N`
- `local_position = (cos(theta), sin(theta), 0)`

This gives exact equal angular spacing and stable index ordering.

### Local +Z hemisphere

Use equal-area Fibonacci sampling over the positive local hemisphere:

- `z = 1 - (k + 0.5)/N`
- `rho = sqrt(1 - z*z)`
- `theta = k*g`
- `local_position = (rho*cos(theta), rho*sin(theta), z)`

The half-sample offset avoids placing a training camera exactly on the equator or pole. Because sampling occurs before applying rig rotation, `+Z` means rig-local +Z.

### Sphere

Use equal-area Fibonacci sampling over the complete sphere:

- `z = 1 - 2*(k + 0.5)/N`
- `rho = sqrt(1 - z*z)`
- `theta = k*g`
- `local_position = (rho*cos(theta), rho*sin(theta), z)`

The half-sample offset avoids exact poles and gives deterministic near-uniform surface coverage.

### Camera orientation

Every camera points its local `-Z` viewing axis at the rig center. Roll is resolved deterministically using rig-local +Z as the preferred up reference and rig-local +Y as the fallback when the preferred vector is parallel or nearly parallel to the viewing direction. The same orientation rule applies to preview objects, rendered poses, and exported matrices.

## Test Trajectories

For an inclusive Blender frame range containing `F` frames, output index `i` uses normalized trajectory parameter:

- `t = 0` when `F == 1`.
- Otherwise, `t = i/(F - 1)`.

The test camera uses the same rig location, rotation, radius, lens, resolution, and pixel aspect as the training cameras.

### Circle test path

- `theta = 2*pi*t`
- `local_position = (cos(theta), sin(theta), 0)`

For more than one frame, the first and last test poses match. This reproduces the closed-orbit endpoint behavior observed in the reference artifact.

### Hemisphere test path

Use a continuous two-turn spiral from the positive pole to the equator:

- `z = 1 - t`
- `rho = sqrt(1 - z*z)`
- `theta = 4*pi*t`
- `local_position = (rho*cos(theta), rho*sin(theta), z)`

### Sphere test path

Use a continuous two-turn spiral from positive to negative pole:

- `z = 1 - 2*t`
- `rho = sqrt(1 - z*z)`
- `theta = 4*pi*t`
- `local_position = (rho*cos(theta), rho*sin(theta), z)`

The path is predetermined rather than user-animated. All rig configuration and shared intrinsics are snapshotted at export start. Training poses remain spatially fixed throughout the export; only the test camera follows its prescribed trajectory while scene animation advances.

## Projection Matrix Convention

For each image, CArr writes a world-to-image projection matrix:

`world_mat_i = K @ [R_cv | t_cv]`

where `K` is the full pinhole intrinsic matrix computed from Blender lens, sensor fit, effective render resolution, resolution percentage, and pixel aspect. This calculation is independent of the shared NGP/NeRF format toggle.

Blender camera coordinates use local `-Z` as forward and local `+Y` as up. Before projection, the inverted Blender camera world matrix is converted to the OpenCV-style camera axes expected by the reference convention: X right, Y down, Z forward. The resulting projection array is explicitly converted to NumPy `float64` and stored with shape `(3, 4)`.

Every `scale_mat_i` is a separate or safely reusable `numpy.eye(4, dtype=numpy.float32)` value. No implicit float promotion may change the stored dtype.

Official Blender builds bundle NumPy, but CArr performs a guarded import when export begins and cancels with an actionable error if an unusual custom Blender build lacks it.

## Output Contract

With both splits enabled, export writes:

```text
<save path>/<sanitized dataset name>/
  cam_train_0/
    rgb/
      000.png
      001.png
      ...
    cameras_sphere.npz
  cam_train_1/
    rgb/
      ...
    cameras_sphere.npz
  ...
  cam_train_<N-1>/
    rgb/
      ...
    cameras_sphere.npz
  cam_test/
    rgb/
      000.png
      001.png
      ...
    cameras_sphere.npz
```

Disabled splits and their directories are omitted. Directory camera indices are not zero-padded, matching the reference (`cam_train_0`). Image names use `format(index, "03d")`; indices beyond 999 naturally grow rather than truncate. NPZ keys use unpadded decimal indices (`world_mat_0`, not `world_mat_000`).

Each enabled camera directory contains exactly `F` `world_mat_i` keys and `F` `scale_mat_i` keys. Training directories repeat their one fixed world matrix. The test directory stores the corresponding trajectory matrix at every index. Matching output indices across directories are the synchronization and timestamp contract; v1 writes no separate time field or JSON manifest.

NPZ entries are inserted in the same order as the reference artifact: all `world_mat_i` keys in ascending index order, followed by all `scale_mat_i` keys in ascending index order.

CArr temporarily sets Blender output to PNG with RGBA color mode and file extensions enabled. It does not force transparent film; alpha values follow the scene's existing film/background configuration. All changed output settings are restored afterward.

No `mask`, `depth`, `transforms_*.json`, `poses_bounds.npy`, or ZIP file is produced. With `Render Frames` disabled, CArr still creates selected camera directories, empty `rgb` directories, and complete NPZ metadata.

## Export Data Flow

1. Validate properties, NumPy availability, output location, and selected splits.
2. Refuse an existing non-empty target directory. An existing empty target is allowed.
3. Ensure the managed rig is complete, then snapshot its configuration and shared intrinsics.
4. Snapshot the active frame, active camera, render filepath, output format settings, and every other scene value CArr will temporarily mutate.
5. Compute all fixed train poses and per-frame test poses without relying on subsequent property or object edits.
6. Create selected output directories and write complete NPZ metadata.
7. If `Render Frames` is disabled, restore scene state, report the directory, and finish.
8. Build a frame-major task queue. For each Blender frame, enqueue all enabled train cameras followed by the enabled test camera.
9. Process one task per modal timer tick, updating Blender progress after each completed image.
10. On success, restore scene state and leave the complete dataset directory in place.

The total render task count is:

`F * ((N if Train else 0) + (1 if Test else 0))`

## Validation and Error Handling

Export cancels before output mutation when:

- Both Train and Test are disabled.
- Radius is not positive.
- Camera count is invalid.
- Save Path is empty or unusable.
- Dataset Name sanitizes to an empty string.
- NumPy cannot be imported.
- The target directory exists and is non-empty.
- The managed cameras cannot be created as perspective cameras.
- The frame range cannot produce at least one frame.

Only one CArr export may run at a time. The operator's poll state disables a second run until the first has finalized.

Metadata preparation and file-writing exceptions are reported before the render queue starts. A render exception or Blender cancellation reports the output camera and Blender frame that failed. Pressing Escape cancels between images.

All success, failure, and cancellation paths restore the original:

- Scene frame.
- Active scene camera.
- Render filepath.
- File format, color mode, and extension setting.
- Any other CArr-mutated render or scene state.

Generated preview objects remain. Partial output remains unarchived for inspection, and the operator clearly reports that it is incomplete. CArr never deletes or overwrites a non-empty pre-existing dataset directory.

## Testing Strategy

### Pure unit tests

Keep deterministic layout, trajectory, and NPZ preparation functions independent enough to test outside Blender where practical.

Geometry tests cover:

- Exact output count and deterministic ordering.
- Unit-length local positions and exact configured world radius.
- Equal circle angles.
- Hemisphere local `z > 0` for train cameras.
- Full-sphere positive and negative Z coverage.
- Correct location and rotation transforms.
- Inward camera direction and stable pole fallback.

Trajectory tests cover:

- Exactly `F` poses for all frame ranges, including `F == 1`.
- Circle first/last equality when `F > 1`.
- Hemisphere local `z` range `[0, 1]`.
- Sphere local `z` range `[-1, 1]`.
- Deterministic two-turn progression.

NeuS writer tests cover:

- Exact NPZ key spelling and count.
- Unpadded NPZ indices and camera-directory indices.
- Minimum three-digit image naming.
- `world_mat_i` shape `(3, 4)` and dtype `float64`.
- `scale_mat_i` shape `(4, 4)`, dtype `float32`, and identity values.
- Repeated train matrices and varying multi-frame test matrices.
- Known Blender camera fixtures producing fixed expected world-to-image matrices.

### Blender integration verification

When a Blender executable is available, headless integration tests or scripts verify:

- Add-on registration and clean unregistration.
- Preview creation, regeneration, deletion, naming, and collection ownership.
- Shared lens and render intrinsics across all generated cameras.
- Timeline-driven test preview with fixed training cameras.
- Frame-major task order and expected image count.
- RGBA PNG output.
- Metadata-only output.
- Train-only and test-only output.
- Existing non-empty directory refusal.
- Render failure and Escape cancellation behavior.
- Restoration of active camera, frame, filepath, and render settings.

If Blender is unavailable in the development environment, these cases form a documented manual verification checklist and must be run in supported Blender 4.2 or later before release. Pure geometry and NPZ tests remain automated.

## Acceptance Criteria

- Default settings create ten fixed training cameras on a circle and one test camera.
- For `F` animation frames with both splits enabled, output contains ten train directories and one test directory, each with `F` RGBA PNG files and one NPZ archive.
- Every NPZ matches the reference key names, matrix shapes, and mixed dtypes.
- Training calibration matrices are constant over time; test matrices follow the selected geometry's prescribed continuous path.
- The same image index across all enabled directories represents the same Blender scene frame.
- Hemisphere cameras occupy the rig-local +Z half before the rig transform is applied.
- Cancelling between images restores Blender state and leaves clearly reported partial output.
- Successful export leaves a usable directory and creates no ZIP, mask folder, depth folder, or alternate-format metadata.
- Existing COS behavior remains unchanged.
