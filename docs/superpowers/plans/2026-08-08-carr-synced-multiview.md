# Camera Array (CArr) Synced Multiview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone Camera Array (CArr) workflow that renders synchronized fixed multiview training cameras plus a deterministic moving test camera into the approved NeuS-derived directory and NPZ convention.

**Architecture:** Keep deterministic geometry and NeuS projection/archive generation in focused modules that can be tested outside Blender. Adapt those functions to managed Blender preview objects in a rig module, then orchestrate validation, metadata generation, and modal frame-major rendering in a separate operator. Register a dedicated CArr panel without changing COS behavior.

**Tech Stack:** Python 3, Blender Python API 4.2+, `mathutils`, NumPy bundled with Blender, pytest 7+, Blender extension manifests, PowerShell test commands.

## Global Constraints

- Blender compatibility floor remains exactly `4.2.0`; Blender 5.1.2 is the primary integration runtime and installed Blender 4.4.0 is the compatibility smoke runtime.
- CArr is a fourth independent method; COS behavior and output remain unchanged.
- Output format is NeuS-derived only; no format selector is added in this iteration.
- Geometry choices are exactly Circle, local +Z Hemisphere, and Sphere.
- Camera Count defaults to 10 and has a hard minimum of 2.
- Train cameras, rig transform, and shared intrinsics are fixed for the whole export; only the deterministic test pose changes by frame.
- Every camera shares the configured lens plus scene resolution, resolution percentage, sensor fit, and pixel aspect.
- Output contains only selected `cam_train_<index>` and `cam_test` directories, `rgb` PNG sequences, optional `log.txt`, and `cameras_sphere.npz` files.
- PNG output is RGBA; CArr does not force transparent film and creates no mask or depth folders.
- `world_mat_i` is `(3, 4)` float64; `scale_mat_i` is `(4, 4)` float32 identity.
- NPZ insertion order is every ascending `world_mat_i`, followed by every ascending `scale_mat_i`.
- Output remains an ordinary directory; no ZIP is created.
- Existing non-empty target directories are never deleted or overwritten.
- Cancellation occurs between images, restores temporary Blender state, and leaves partial output inspectable.
- Tests and docs must not be included in the built Blender extension package.

---

## File Structure

### New runtime files

- `carr_geometry.py`: pure deterministic training layouts, test trajectories, rig transforms, and inward-looking camera matrices.
- `carr_neus.py`: guarded NumPy access, full pinhole intrinsics, Blender-to-OpenCV projection conversion, image naming, and typed NPZ writing.
- `carr_rig.py`: CArr-managed Blender collection, camera object lifecycle, pose adaptation, property callbacks, and frame-change preview handler.
- `carr_operator.py`: validation, safe directory preparation, metadata/log creation, render-task construction, modal rendering, cancellation, and state restoration.
- `carr_ui.py`: the dedicated `Camera Array CArr` panel.

### New tests

- `tests/test_carr_geometry.py`: pure geometry, trajectory, transform, and orientation tests.
- `tests/test_carr_neus.py`: pure intrinsics, projection, dtype, ordering, naming, and archive tests.
- `tests/blender/carr_test_support.py`: reusable Blender test bootstrap and cleanup helpers.
- `tests/blender/test_carr_preview.py`: registration and managed-preview integration checks.
- `tests/blender/test_carr_metadata_export.py`: metadata-only export, split selection, logging, collision, and state checks.
- `tests/blender/test_carr_render.py`: task order, PNG RGBA rendering, operator registration, and state restoration checks.
- `tests/blender/test_carr_registration.py`: final panel, property, handler, operator, and version smoke checks.
- `tests/blender/manual_carr_modal_setup.py`: tiny interactive scene used to verify visible progress and Escape cancellation.

### Modified files

- `__init__.py:2, 5-12, 60-106, 109-135`: import/register CArr modules, properties, operator, UI, and handler; bump add-on version.
- `README.md:42-139`: document four methods, CArr controls, ignored shared controls, output structure, trajectory behavior, directory output, cancellation, and NeuS compatibility.
- `blender_manifest.toml:3, 18-26`: bump the extension version and exclude `/tests/` from packages.

No changes are planned for `cos_operator.py`, `cos_ui.py`, or COS helper behavior.

---

### Task 1: Deterministic Train Layouts and Test Trajectories

**Files:**
- Create: `carr_geometry.py`
- Create: `tests/test_carr_geometry.py`

**Interfaces:**
- Consumes: geometry names `CIRCLE`, `HEMISPHERE`, `SPHERE`; zero-based camera or output indices.
- Produces: `train_local_positions(geometry: str, count: int) -> list[Vector3]` and `test_local_position(geometry: str, index: int, frame_count: int) -> Vector3`, where `Vector3 = tuple[float, float, float]`.

- [ ] **Step 1: Write failing tests for the three training layouts**

```python
import math

import pytest

import carr_geometry


def length(vector):
    return math.sqrt(sum(component * component for component in vector))


def test_circle_has_exact_count_unit_radius_and_equal_angles():
    points = carr_geometry.train_local_positions(carr_geometry.CIRCLE, 4)
    assert points == pytest.approx([
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
    ])


def test_hemisphere_is_deterministic_unit_length_and_positive_z():
    first = carr_geometry.train_local_positions(carr_geometry.HEMISPHERE, 10)
    second = carr_geometry.train_local_positions(carr_geometry.HEMISPHERE, 10)
    assert first == second
    assert all(length(point) == pytest.approx(1.0) for point in first)
    assert all(0.0 < point[2] < 1.0 for point in first)


def test_sphere_is_deterministic_and_covers_both_z_halves():
    points = carr_geometry.train_local_positions(carr_geometry.SPHERE, 10)
    assert all(length(point) == pytest.approx(1.0) for point in points)
    assert any(point[2] > 0.0 for point in points)
    assert any(point[2] < 0.0 for point in points)


@pytest.mark.parametrize('count', [0, 1])
def test_training_layout_rejects_fewer_than_two_cameras(count):
    with pytest.raises(ValueError, match='at least 2'):
        carr_geometry.train_local_positions(carr_geometry.CIRCLE, count)
```

- [ ] **Step 2: Run the training-layout tests and verify the missing module failure**

Run: `python -m pytest tests/test_carr_geometry.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'carr_geometry'`.

- [ ] **Step 3: Implement deterministic train layouts**

```python
import math


CIRCLE = 'CIRCLE'
HEMISPHERE = 'HEMISPHERE'
SPHERE = 'SPHERE'
GEOMETRIES = (CIRCLE, HEMISPHERE, SPHERE)
GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


def _surface_point(z, theta):
    rho = math.sqrt(max(0.0, 1.0 - z * z))
    return (rho * math.cos(theta), rho * math.sin(theta), z)


def train_local_positions(geometry, count):
    if geometry not in GEOMETRIES:
        raise ValueError('Unknown CArr geometry: {}'.format(geometry))
    if count < 2:
        raise ValueError('CArr requires at least 2 training cameras')
    points = []
    for index in range(count):
        if geometry == CIRCLE:
            theta = 2.0 * math.pi * index / count
            point = (math.cos(theta), math.sin(theta), 0.0)
        elif geometry == HEMISPHERE:
            point = _surface_point(1.0 - (index + 0.5) / count, index * GOLDEN_ANGLE)
        else:
            point = _surface_point(1.0 - 2.0 * (index + 0.5) / count, index * GOLDEN_ANGLE)
        points.append(point)
    return points
```

- [ ] **Step 4: Run the training-layout tests**

Run: `python -m pytest tests/test_carr_geometry.py -v`

Expected: 5 tests PASS.

- [ ] **Step 5: Add failing tests for all test paths**

```python
def test_circle_test_path_is_closed():
    first = carr_geometry.test_local_position(carr_geometry.CIRCLE, 0, 5)
    middle = carr_geometry.test_local_position(carr_geometry.CIRCLE, 2, 5)
    last = carr_geometry.test_local_position(carr_geometry.CIRCLE, 4, 5)
    assert first == pytest.approx(last)
    assert middle != pytest.approx(first)


def test_hemisphere_path_runs_from_positive_pole_to_equator():
    points = [carr_geometry.test_local_position(carr_geometry.HEMISPHERE, index, 5) for index in range(5)]
    assert points[0] == pytest.approx((0.0, 0.0, 1.0))
    assert points[-1][2] == pytest.approx(0.0)
    assert all(0.0 <= point[2] <= 1.0 for point in points)


def test_sphere_path_runs_from_positive_to_negative_pole():
    points = [carr_geometry.test_local_position(carr_geometry.SPHERE, index, 5) for index in range(5)]
    assert points[0] == pytest.approx((0.0, 0.0, 1.0))
    assert points[-1] == pytest.approx((0.0, 0.0, -1.0))


def test_single_frame_uses_trajectory_parameter_zero():
    assert carr_geometry.test_local_position(carr_geometry.CIRCLE, 0, 1) == pytest.approx((1.0, 0.0, 0.0))
    assert carr_geometry.test_local_position(carr_geometry.HEMISPHERE, 0, 1) == pytest.approx((0.0, 0.0, 1.0))
    assert carr_geometry.test_local_position(carr_geometry.SPHERE, 0, 1) == pytest.approx((0.0, 0.0, 1.0))
```

- [ ] **Step 6: Run the trajectory tests and verify the missing function**

Run: `python -m pytest tests/test_carr_geometry.py -v`

Expected: FAIL because `test_local_position` is absent.

- [ ] **Step 7: Implement the approved trajectories**

```python
def test_local_position(geometry, index, frame_count):
    if geometry not in GEOMETRIES:
        raise ValueError('Unknown CArr geometry: {}'.format(geometry))
    if frame_count < 1:
        raise ValueError('CArr requires at least 1 animation frame')
    if index < 0 or index >= frame_count:
        raise ValueError('CArr output index is outside the animation range')
    time = 0.0 if frame_count == 1 else index / (frame_count - 1)
    if geometry == CIRCLE:
        theta = 2.0 * math.pi * time
        return (math.cos(theta), math.sin(theta), 0.0)
    if geometry == HEMISPHERE:
        return _surface_point(1.0 - time, 4.0 * math.pi * time)
    return _surface_point(1.0 - 2.0 * time, 4.0 * math.pi * time)
```

- [ ] **Step 8: Run all Task 1 tests and commit**

```powershell
python -m pytest tests/test_carr_geometry.py -v
git add carr_geometry.py tests/test_carr_geometry.py
git commit -m "feat: add deterministic CArr sampling"
```

Expected: 9 tests PASS and the commit succeeds.

---

### Task 2: Rig Transforms and Inward Camera Orientation

**Files:**
- Modify: `carr_geometry.py`
- Modify: `tests/test_carr_geometry.py`

**Interfaces:**
- Consumes: `Vector3` values from Task 1, a row-major 3x3 rotation, rig center, and positive radius.
- Produces: `transform_position`, `transform_direction`, and `look_at_matrix(camera_position, target, preferred_up, fallback_up) -> Matrix4`.

- [ ] **Step 1: Add failing transform and orientation tests**

```python
def test_transform_position_applies_radius_rotation_then_translation():
    rotation = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    actual = carr_geometry.transform_position((1.0, 0.0, 0.0), (10.0, 20.0, 30.0), rotation, 4.0)
    assert actual == pytest.approx((10.0, 24.0, 30.0))


def test_look_at_points_local_negative_z_at_target():
    matrix = carr_geometry.look_at_matrix(
        (4.0, 0.0, 0.0), (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0), (0.0, 1.0, 0.0),
    )
    assert tuple(-matrix[row][2] for row in range(3)) == pytest.approx((-1.0, 0.0, 0.0))
    assert tuple(matrix[row][3] for row in range(3)) == pytest.approx((4.0, 0.0, 0.0))


def test_look_at_uses_fallback_at_positive_pole():
    matrix = carr_geometry.look_at_matrix(
        (0.0, 0.0, 4.0), (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0), (0.0, 1.0, 0.0),
    )
    assert tuple(-matrix[row][2] for row in range(3)) == pytest.approx((0.0, 0.0, -1.0))
    assert length(tuple(matrix[row][0] for row in range(3))) == pytest.approx(1.0)
```

- [ ] **Step 2: Run the focused tests and verify missing APIs**

Run: `python -m pytest tests/test_carr_geometry.py -k "transform or look_at" -v`

Expected: FAIL because the functions are absent.

- [ ] **Step 3: Implement vector, transform, and look-at math**

```python
def _dot(left, right):
    return sum(left[index] * right[index] for index in range(3))


def _cross(left, right):
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _scale(vector, scalar):
    return tuple(component * scalar for component in vector)


def _normalize(vector):
    magnitude = math.sqrt(_dot(vector, vector))
    if magnitude <= 1.0e-12:
        raise ValueError('Cannot normalize a zero-length vector')
    return _scale(vector, 1.0 / magnitude)


def transform_direction(direction, rotation_rows):
    return tuple(_dot(row, direction) for row in rotation_rows)


def transform_position(point, location, rotation_rows, radius):
    if radius <= 0.0:
        raise ValueError('CArr radius must be positive')
    rotated = transform_direction(_scale(point, radius), rotation_rows)
    return tuple(location[index] + rotated[index] for index in range(3))


def look_at_matrix(camera_position, target, preferred_up, fallback_up):
    forward = _normalize(tuple(target[i] - camera_position[i] for i in range(3)))
    up = _normalize(preferred_up)
    if abs(_dot(forward, up)) >= 1.0 - 1.0e-6:
        up = _normalize(fallback_up)
    right = _normalize(_cross(forward, up))
    corrected_up = _normalize(_cross(right, forward))
    local_z = _scale(forward, -1.0)
    return (
        (right[0], corrected_up[0], local_z[0], camera_position[0]),
        (right[1], corrected_up[1], local_z[1], camera_position[1]),
        (right[2], corrected_up[2], local_z[2], camera_position[2]),
        (0.0, 0.0, 0.0, 1.0),
    )
```

- [ ] **Step 4: Run all geometry tests and commit**

```powershell
python -m pytest tests/test_carr_geometry.py -v
git add carr_geometry.py tests/test_carr_geometry.py
git commit -m "feat: add CArr rig pose math"
```

Expected: 12 tests PASS and the commit succeeds.

---

### Task 3: NeuS Projection and Typed NPZ Writer

**Files:**
- Create: `carr_neus.py`
- Create: `tests/test_carr_neus.py`

**Interfaces:**
- Consumes: raw camera/render scalar settings and row-major 4x4 camera world matrices.
- Produces: `intrinsic_matrix`, `world_projection`, `archive_entries`, `write_camera_archive`, `image_filename`, and `numpy_available`.

- [ ] **Step 1: Write failing naming, intrinsics, and reference-projection tests**

```python
import numpy as np
import pytest

import carr_neus


def test_image_filename_has_minimum_three_digits():
    assert carr_neus.image_filename(0) == '000.png'
    assert carr_neus.image_filename(99) == '099.png'
    assert carr_neus.image_filename(1000) == '1000.png'


def test_horizontal_intrinsics():
    actual = carr_neus.intrinsic_matrix(
        50.0, 10.0, 10.0, 'HORIZONTAL', 512, 512, 100.0, 1.0, 1.0
    )
    expected = np.array([[2560.0, 0.0, 256.0], [0.0, 2560.0, 256.0], [0.0, 0.0, 1.0]])
    assert actual == pytest.approx(expected)


def test_vertical_and_auto_portrait_intrinsics_match():
    arguments = (50.0, 36.0, 20.0, 400, 800, 100.0, 1.0, 1.0)
    vertical = carr_neus.intrinsic_matrix(
        arguments[0], arguments[1], arguments[2], 'VERTICAL', *arguments[3:]
    )
    automatic = carr_neus.intrinsic_matrix(
        arguments[0], arguments[1], arguments[2], 'AUTO', *arguments[3:]
    )
    expected = np.array([[1000.0, 0.0, 200.0], [0.0, 1000.0, 400.0], [0.0, 0.0, 1.0]])
    assert vertical == pytest.approx(expected)
    assert automatic == pytest.approx(expected)


def test_missing_numpy_is_reported_at_call_time(monkeypatch):
    monkeypatch.setattr(carr_neus, '_numpy', None)
    assert carr_neus.numpy_available() is False
    with pytest.raises(RuntimeError, match='requires NumPy'):
        carr_neus.image_filename(0) if carr_neus.numpy_available() else carr_neus._require_numpy()


def test_projection_matches_reference_cam_test_world_mat_zero():
    intrinsics = np.array([[2560.0, 0.0, 256.0], [0.0, 2560.0, 256.0], [0.0, 0.0, 1.0]])
    camera_world = np.eye(4)
    camera_world[2, 3] = 10.0
    expected = np.array([
        [2560.0, 0.0, -256.0, 2560.0],
        [0.0, -2560.0, -256.0, 2560.0],
        [0.0, 0.0, -1.0, 10.0],
    ])
    actual = carr_neus.world_projection(camera_world, intrinsics)
    assert actual.shape == (3, 4)
    assert actual.dtype == np.float64
    assert actual == pytest.approx(expected)
```

- [ ] **Step 2: Run tests and verify the missing module failure**

Run: `python -m pytest tests/test_carr_neus.py -v`

Expected: FAIL during collection because `carr_neus` is absent.

- [ ] **Step 3: Implement guarded NumPy, naming, intrinsics, and projection**

```python
try:
    import numpy as _numpy
except ImportError:
    _numpy = None


def numpy_available():
    return _numpy is not None


def _require_numpy():
    if _numpy is None:
        raise RuntimeError('CArr NeuS export requires NumPy in Blender Python')
    return _numpy


def image_filename(index):
    if index < 0:
        raise ValueError('CArr image index cannot be negative')
    return '{:03d}.png'.format(index)


def intrinsic_matrix(lens, sensor_width, sensor_height, sensor_fit, res_x, res_y, percentage, aspect_x, aspect_y):
    np = _require_numpy()
    width = res_x * percentage / 100.0
    height = res_y * percentage / 100.0
    ratio = aspect_x / aspect_y
    fit = sensor_fit
    if fit == 'AUTO':
        fit = 'VERTICAL' if width < height or (width == height and aspect_x * width <= aspect_y * height) else 'HORIZONTAL'
    if fit == 'HORIZONTAL':
        fl_x = lens / sensor_width * width
        fl_y = lens / sensor_width * width * ratio
    elif fit == 'VERTICAL':
        sensor_size = sensor_height if width <= height else sensor_width
        fl_x = lens / sensor_size * width / ratio
        fl_y = lens / sensor_size * width
    else:
        raise ValueError('Unsupported Blender sensor fit: {}'.format(sensor_fit))
    return np.array([[fl_x, 0.0, width / 2.0], [0.0, fl_y, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def world_projection(camera_world, intrinsics):
    np = _require_numpy()
    camera_world = np.asarray(camera_world, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if camera_world.shape != (4, 4) or intrinsics.shape != (3, 3):
        raise ValueError('CArr projection requires 4x4 world and 3x3 intrinsic matrices')
    blender_to_cv = np.diag([1.0, -1.0, -1.0, 1.0])
    return np.asarray(intrinsics @ (blender_to_cv @ np.linalg.inv(camera_world))[:3, :], dtype=np.float64)
```

- [ ] **Step 4: Run the first NeuS tests**

Run: `python -m pytest tests/test_carr_neus.py -v`

Expected: 5 tests PASS.

- [ ] **Step 5: Add failing archive ordering and mixed-dtype round-trip test**

```python
def test_archive_matches_reference_order_shapes_and_dtypes(tmp_path):
    worlds = [np.full((3, 4), index, dtype=np.float64) for index in range(3)]
    entries = carr_neus.archive_entries(worlds)
    expected_keys = [
        'world_mat_0', 'world_mat_1', 'world_mat_2',
        'scale_mat_0', 'scale_mat_1', 'scale_mat_2',
    ]
    assert list(entries) == expected_keys
    assert entries['world_mat_0'].dtype == np.float64
    assert entries['scale_mat_0'].shape == (4, 4)
    assert entries['scale_mat_0'].dtype == np.float32
    assert entries['scale_mat_0'] == pytest.approx(np.eye(4, dtype=np.float32))
    output = tmp_path / 'cameras_sphere.npz'
    carr_neus.write_camera_archive(output, worlds)
    loaded = np.load(output)
    assert loaded.files == expected_keys
    assert loaded['world_mat_2'] == pytest.approx(worlds[2])
```

- [ ] **Step 6: Run the archive test and verify missing APIs**

Run: `python -m pytest tests/test_carr_neus.py -k archive -v`

Expected: FAIL because `archive_entries` is absent.

- [ ] **Step 7: Implement ordered mixed-dtype archives**

```python
def archive_entries(world_matrices):
    np = _require_numpy()
    worlds = [np.asarray(matrix, dtype=np.float64) for matrix in world_matrices]
    if not worlds or any(matrix.shape != (3, 4) for matrix in worlds):
        raise ValueError('CArr archives require one or more 3x4 world matrices')
    entries = {}
    for index, matrix in enumerate(worlds):
        entries['world_mat_{}'.format(index)] = matrix
    for index in range(len(worlds)):
        entries['scale_mat_{}'.format(index)] = np.eye(4, dtype=np.float32)
    return entries


def write_camera_archive(path, world_matrices):
    _require_numpy().savez(path, **archive_entries(world_matrices))
```

- [ ] **Step 8: Run all pure tests and commit**

```powershell
python -m pytest tests/test_carr_geometry.py tests/test_carr_neus.py -v
git add carr_neus.py tests/test_carr_neus.py
git commit -m "feat: add NeuS camera archive writer"
```

Expected: all pure tests PASS under NumPy 1.26.4 and the commit succeeds.

---

### Task 4: Managed Blender Preview Rig

**Files:**
- Create: `carr_rig.py`
- Create: `tests/blender/carr_test_support.py`
- Create: `tests/blender/test_carr_preview.py`
- Modify: `__init__.py:2, 60-96, 109-135`

**Interfaces:**
- Consumes: all `carr_geometry` functions from Tasks 1-2.
- Produces: `CARR_COLLECTION_NAME`, `CARR_RIG_NAME`, `CARR_CAMERA_DATA_NAME`, `CARR_TRAIN_PREFIX`, `CARR_TEST_NAME`, `ensure_preview(context)`, `remove_preview(scene)`, `train_camera_matrices(scene)`, `test_camera_matrix(scene, output_index, frame_count)`, property callbacks, and persistent `carr_frame_change(scene)`.

- [ ] **Step 1: Create the reusable Blender test bootstrap**

```python
from contextlib import contextmanager
from pathlib import Path
import sys

import bpy


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))


@contextmanager
def registered_addon():
    import BlenderNeRF
    BlenderNeRF.register()
    try:
        yield BlenderNeRF
    finally:
        scene = bpy.context.scene
        if hasattr(scene, 'carr_show_rig'):
            scene.carr_show_rig = False
        BlenderNeRF.unregister()


def matrix_rows(matrix):
    return tuple(tuple(float(value) for value in row) for row in matrix)
```

- [ ] **Step 2: Write a failing Blender preview integration script**

```python
import math

import bpy

from carr_test_support import matrix_rows, registered_addon


with registered_addon() as addon:
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 5
    scene.carr_geometry = 'CIRCLE'
    scene.carr_camera_count = 3
    scene.carr_location = (0.0, 0.0, 0.0)
    scene.carr_rotation = (0.0, 0.0, 0.0)
    scene.carr_radius = 4.0
    scene.carr_focal = 50.0
    scene.carr_show_rig = True

    collection = bpy.data.collections[addon.carr_rig.CARR_COLLECTION_NAME]
    train = sorted(
        [obj for obj in collection.objects if obj.name.startswith(addon.carr_rig.CARR_TRAIN_PREFIX)],
        key=lambda obj: obj.name,
    )
    test = collection.objects[addon.carr_rig.CARR_TEST_NAME]
    assert len(train) == 3
    assert all(obj.type == 'CAMERA' for obj in train + [test])
    assert all(obj.data is train[0].data for obj in train + [test])
    assert train[0].data.lens == 50.0
    assert all(math.isclose(obj.location.length, 4.0, rel_tol=1.0e-6) for obj in train)

    fixed_before = [matrix_rows(obj.matrix_world) for obj in train]
    scene.frame_set(1)
    test_start = matrix_rows(test.matrix_world)
    scene.frame_set(3)
    test_middle = matrix_rows(test.matrix_world)
    assert [matrix_rows(obj.matrix_world) for obj in train] == fixed_before
    assert test_start != test_middle

    bpy.data.objects.remove(test, do_unlink=True)
    addon.carr_rig.ensure_preview(bpy.context)
    collection = bpy.data.collections[addon.carr_rig.CARR_COLLECTION_NAME]
    assert addon.carr_rig.CARR_TEST_NAME in collection.objects

    scene.carr_camera_count = 4
    train = [obj for obj in collection.objects if obj.name.startswith(addon.carr_rig.CARR_TRAIN_PREFIX)]
    assert len(train) == 4
    scene.carr_show_rig = False
    assert addon.carr_rig.CARR_COLLECTION_NAME not in bpy.data.collections

print('CArr preview checks passed')
```

- [ ] **Step 3: Run the preview script under Blender 5.1 and verify failure**

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --background --factory-startup --python-exit-code 1 --python 'tests/blender/test_carr_preview.py'
```

Expected: non-zero exit because CArr properties and `carr_rig` are absent.

- [ ] **Step 4: Implement the Blender pose adapter and managed lifecycle**

Create `carr_rig.py` with these constants and pose boundary:

```python
import bpy
from bpy.app.handlers import persistent
from mathutils import Euler, Matrix

from . import carr_geometry


CARR_COLLECTION_NAME = 'BlenderNeRF CArr'
CARR_RIG_NAME = 'BlenderNeRF CArr Rig'
CARR_CAMERA_DATA_NAME = 'BlenderNeRF CArr Camera'
CARR_TRAIN_PREFIX = 'BlenderNeRF CArr Train '
CARR_TEST_NAME = 'BlenderNeRF CArr Test'
MANAGED_KEY = 'blendernerf_carr_managed'


def _rotation_rows(scene):
    rotation = Euler(scene.carr_rotation).to_matrix()
    return tuple(tuple(float(value) for value in row) for row in rotation)


def _pose_matrix(scene, local_position):
    rotation_rows = _rotation_rows(scene)
    center = tuple(scene.carr_location)
    position = carr_geometry.transform_position(local_position, center, rotation_rows, scene.carr_radius)
    preferred_up = carr_geometry.transform_direction((0.0, 0.0, 1.0), rotation_rows)
    fallback_up = carr_geometry.transform_direction((0.0, 1.0, 0.0), rotation_rows)
    return Matrix(carr_geometry.look_at_matrix(position, center, preferred_up, fallback_up))


def train_camera_matrices(scene):
    points = carr_geometry.train_local_positions(scene.carr_geometry, scene.carr_camera_count)
    return [_pose_matrix(scene, point) for point in points]


def test_camera_matrix(scene, output_index, frame_count):
    point = carr_geometry.test_local_position(scene.carr_geometry, output_index, frame_count)
    return _pose_matrix(scene, point)
```

Implement `ensure_preview(context)` with this exact behavior:

1. Reuse or create the dedicated collection under `scene.collection`.
2. Remove only objects in that collection carrying `MANAGED_KEY = True`.
3. Create the mount empty, one shared perspective camera datablock, exactly `scene.carr_camera_count` train objects, and one test object.
4. Name train objects with `'{:03d}'.format(index)` and assign Task 2 matrices.
5. Set the shared datablock lens to `scene.carr_focal` and mark every generated datablock/object.
6. Set the test matrix from the current frame's zero-based output index.
7. Return train objects in numeric order and the test object.

Implement `remove_preview(scene)` so it removes marked objects, removes the collection when empty, and removes the shared camera datablock when it has zero users. Implement callbacks and the handler as:

```python
def carr_show_rig_update(scene, context):
    if scene.carr_show_rig:
        ensure_preview(context)
    else:
        remove_preview(scene)


def carr_rig_property_update(scene, context):
    if scene.carr_show_rig:
        ensure_preview(context)


@persistent
def carr_frame_change(scene):
    if not scene.carr_show_rig or CARR_TEST_NAME not in scene.objects:
        return
    frame_count = scene.frame_end - scene.frame_start + 1
    output_index = scene.frame_current - scene.frame_start
    scene.objects[CARR_TEST_NAME].matrix_world = test_camera_matrix(scene, output_index, frame_count)
```

- [ ] **Step 5: Register the CArr properties and handler**

Import `carr_rig` in `__init__.py`. Add these properties after COS properties:

```python
('carr_dataset_name', bpy.props.StringProperty(name='Name', description='Name of the CArr dataset directory', default='dataset')),
('carr_geometry', bpy.props.EnumProperty(
    name='Geometry',
    items=(
        ('CIRCLE', 'Circle', 'Equally spaced cameras on the local XY circle'),
        ('HEMISPHERE', 'Hemisphere', 'Near-uniform cameras on the local +Z hemisphere'),
        ('SPHERE', 'Sphere', 'Near-uniform cameras on the full sphere'),
    ),
    default='CIRCLE', update=carr_rig.carr_rig_property_update,
)),
('carr_camera_count', bpy.props.IntProperty(
    name='Camera Count', default=10, min=2,
    description='Number of fixed synchronized training cameras',
    update=carr_rig.carr_rig_property_update,
)),
('carr_location', bpy.props.FloatVectorProperty(
    name='Location', unit='LENGTH', update=carr_rig.carr_rig_property_update,
)),
('carr_rotation', bpy.props.FloatVectorProperty(
    name='Rotation', unit='ROTATION', subtype='EULER', update=carr_rig.carr_rig_property_update,
)),
('carr_radius', bpy.props.FloatProperty(
    name='Radius', default=4.0, min=0.001, unit='LENGTH', update=carr_rig.carr_rig_property_update,
)),
('carr_focal', bpy.props.FloatProperty(
    name='Lens', default=50.0, min=1.0, max=5000.0, unit='CAMERA', update=carr_rig.carr_rig_property_update,
)),
('carr_show_rig', bpy.props.BoolProperty(
    name='Show Rig', default=False, update=carr_rig.carr_show_rig_update,
)),
```

Append `carr_rig.carr_frame_change` in `register()` only when absent. Remove it in `unregister()` only when present.

- [ ] **Step 6: Run preview checks on Blender 5.1 and 4.4**

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --background --factory-startup --python-exit-code 1 --python 'tests/blender/test_carr_preview.py'
& 'C:\Program Files\Blender Foundation\Blender 4.4\blender.exe' --background --factory-startup --python-exit-code 1 --python 'tests/blender/test_carr_preview.py'
```

Expected: both commands exit 0 with `CArr preview checks passed`.

- [ ] **Step 7: Run pure regressions and commit**

```powershell
python -m pytest tests/test_carr_geometry.py tests/test_carr_neus.py -v
git add carr_rig.py __init__.py tests/blender/carr_test_support.py tests/blender/test_carr_preview.py
git commit -m "feat: add managed CArr preview rig"
```

Expected: all tests PASS and the commit succeeds.

---

### Task 5: Safe Metadata-Only Export Preparation

**Files:**
- Create: `carr_operator.py`
- Create: `tests/blender/test_carr_metadata_export.py`

**Interfaces:**
- Consumes: `carr_rig.ensure_preview`, `carr_rig.test_camera_matrix`, and all `carr_neus` public functions.
- Produces: `CArrValidationError`, immutable `RenderCalibration`, immutable `RenderTask`, mutable `PreparedExport`, `validate_export(scene)`, `prepare_export(context)`, `restore_scene(scene, initial_state)`, and `complete_metadata_only(context, prepared)`.

- [ ] **Step 1: Write the failing metadata-only integration script**

```python
from pathlib import Path
import tempfile

import bpy
import numpy as np

from carr_test_support import registered_addon


with registered_addon() as addon, tempfile.TemporaryDirectory() as temporary:
    from BlenderNeRF import carr_operator
    scene = bpy.context.scene
    scene.frame_start = 10
    scene.frame_end = 12
    scene.frame_set(11)
    original = (scene.frame_current, scene.camera, scene.render.filepath)
    scene.save_path = temporary
    scene.carr_dataset_name = 'synced array'
    scene.carr_geometry = 'CIRCLE'
    scene.carr_camera_count = 2
    scene.train_data = True
    scene.test_data = True
    scene.render_frames = False
    scene.logs = True

    prepared = carr_operator.prepare_export(bpy.context)
    assert carr_operator.complete_metadata_only(bpy.context, prepared) == {'FINISHED'}
    root = Path(temporary) / 'synced_array'
    assert sorted(path.name for path in root.iterdir()) == ['cam_test', 'cam_train_0', 'cam_train_1', 'log.txt']
    expected_keys = [
        'world_mat_0', 'world_mat_1', 'world_mat_2',
        'scale_mat_0', 'scale_mat_1', 'scale_mat_2',
    ]
    for camera_name in ('cam_train_0', 'cam_train_1', 'cam_test'):
        camera_root = root / camera_name
        assert (camera_root / 'rgb').is_dir()
        assert list((camera_root / 'rgb').iterdir()) == []
        assert np.load(camera_root / 'cameras_sphere.npz').files == expected_keys
    train = np.load(root / 'cam_train_0' / 'cameras_sphere.npz')
    test = np.load(root / 'cam_test' / 'cameras_sphere.npz')
    assert np.allclose(train['world_mat_0'], train['world_mat_2'])
    assert not np.allclose(test['world_mat_0'], test['world_mat_1'])
    assert train['world_mat_0'].dtype == np.float64
    assert train['scale_mat_0'].dtype == np.float32
    assert (scene.frame_current, scene.camera, scene.render.filepath) == original

    assert not Path(str(root) + '.zip').exists()
    assert not any(root.glob('*/mask'))
    assert not any(root.glob('*/depth'))
    log = (root / 'log.txt').read_text(encoding='utf-8')
    assert '"Method": "CArr"' in log

    scene.logs = False
    scene.carr_dataset_name = 'train only'
    scene.train_data = True
    scene.test_data = False
    train_only = carr_operator.prepare_export(bpy.context)
    carr_operator.complete_metadata_only(bpy.context, train_only)
    train_only_root = Path(temporary) / 'train_only'
    assert (train_only_root / 'cam_train_0').is_dir()
    assert not (train_only_root / 'cam_test').exists()

    scene.carr_dataset_name = 'test only'
    scene.train_data = False
    scene.test_data = True
    test_only = carr_operator.prepare_export(bpy.context)
    carr_operator.complete_metadata_only(bpy.context, test_only)
    test_only_root = Path(temporary) / 'test_only'
    assert (test_only_root / 'cam_test').is_dir()
    assert not any(test_only_root.glob('cam_train_*'))

    empty_root = Path(temporary) / 'empty_target'
    empty_root.mkdir()
    scene.carr_dataset_name = 'empty target'
    empty_target = carr_operator.prepare_export(bpy.context)
    carr_operator.complete_metadata_only(bpy.context, empty_target)
    assert (empty_root / 'cam_test' / 'cameras_sphere.npz').is_file()

    scene.carr_dataset_name = 'synced array'
    scene.train_data = True
    scene.test_data = True
    try:
        carr_operator.prepare_export(bpy.context)
    except carr_operator.CArrValidationError as exception:
        assert 'non-empty' in str(exception)
    else:
        raise AssertionError('CArr accepted an existing non-empty directory')

print('CArr metadata export checks passed')
```

- [ ] **Step 2: Run the metadata script and verify the missing module failure**

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --background --factory-startup --python-exit-code 1 --python 'tests/blender/test_carr_metadata_export.py'
```

Expected: non-zero exit because `carr_operator.py` is absent.

- [ ] **Step 3: Define export data types and validation**

```python
from dataclasses import dataclass
import datetime
import json
import os
from pathlib import Path

import bpy

from . import carr_neus, carr_rig


class CArrValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class RenderCalibration:
    lens: float
    sensor_width: float
    sensor_height: float
    sensor_fit: str
    resolution_x: int
    resolution_y: int
    resolution_percentage: int
    pixel_aspect_x: float
    pixel_aspect_y: float


@dataclass(frozen=True)
class RenderTask:
    split: str
    camera: object
    blender_frame: int
    output_index: int
    filepath: str
    matrix_world: object


@dataclass
class PreparedExport:
    output_path: str
    frames: list
    tasks: list
    initial_state: dict
    train_cameras: list
    test_camera: object
    calibration: RenderCalibration
```

Implement `validate_export(scene)` to raise one actionable `CArrValidationError` for: neither split selected, count below 2, non-positive radius, empty save path, empty sanitized name, unavailable NumPy, reversed frame range, unusable target path, or existing non-empty target. Return the sanitized output path and inclusive frame list only after all checks pass.

- [ ] **Step 4: Implement projection snapshots, output creation, tasks, and log**

Add `_calibration(scene, camera)` that snapshots lens, sensor width/height/fit, render resolution/percentage, and pixel aspect into `RenderCalibration`. Add `_intrinsics(calibration)` that passes those exact values to `carr_neus.intrinsic_matrix`. Add `_projection(matrix_world, intrinsics)` that converts `mathutils.Matrix` to float rows and calls `carr_neus.world_projection`.

Implement `prepare_export(context)` in this order:

1. Validate before creating output.
2. Set `scene.carr_show_rig = True` and call `carr_rig.ensure_preview(context)` so the rig remains after export.
3. Snapshot frame, active camera, render filepath, file format, color mode, `use_file_extension`, shared camera lens/sensor settings, resolution, resolution percentage, and pixel aspect.
4. Copy train matrices once; compute every test matrix by zero-based output index.
5. Create only selected camera and empty `rgb` directories.
6. Write repeated train projections and per-index test projections.
7. Build frame-major tasks: numeric train order followed by test for each Blender frame.
8. When logs are enabled, write root `log.txt` JSON containing version, date/time, Train, Test, Render Frames, Save Path, Method=`CArr`, Geometry, Camera Count, Location, Rotation, Radius, Lens, Frame Count, and Dataset Name.
9. Restore captured state and re-raise on any exception; never delete partial files.
10. Return `PreparedExport` without changing the current frame or active camera.

- [ ] **Step 5: Implement restoration and metadata completion**

```python
def restore_scene(scene, initial_state):
    scene.render.filepath = initial_state['render_filepath']
    scene.render.image_settings.file_format = initial_state['file_format']
    scene.render.image_settings.color_mode = initial_state['color_mode']
    scene.render.use_file_extension = initial_state['use_file_extension']
    scene.render.resolution_x = initial_state['resolution_x']
    scene.render.resolution_y = initial_state['resolution_y']
    scene.render.resolution_percentage = initial_state['resolution_percentage']
    scene.render.pixel_aspect_x = initial_state['pixel_aspect_x']
    scene.render.pixel_aspect_y = initial_state['pixel_aspect_y']
    if carr_rig.CARR_TEST_NAME in scene.objects:
        camera_data = scene.objects[carr_rig.CARR_TEST_NAME].data
        camera_data.lens = initial_state['lens']
        camera_data.sensor_width = initial_state['sensor_width']
        camera_data.sensor_height = initial_state['sensor_height']
        camera_data.sensor_fit = initial_state['sensor_fit']
    scene.camera = initial_state['camera']
    scene.frame_set(initial_state['frame'])


def complete_metadata_only(context, prepared):
    restore_scene(context.scene, prepared.initial_state)
    return {'FINISHED'}
```

- [ ] **Step 6: Run metadata and pure checks on both NumPy generations**

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --background --factory-startup --python-exit-code 1 --python 'tests/blender/test_carr_metadata_export.py'
& 'C:\Program Files\Blender Foundation\Blender 4.4\blender.exe' --background --factory-startup --python-exit-code 1 --python 'tests/blender/test_carr_metadata_export.py'
python -m pytest tests/test_carr_geometry.py tests/test_carr_neus.py -v
```

Expected: all commands PASS; stored dtypes remain float64/float32 with Blender 5.1 NumPy 2.3.4, Blender 4.4 NumPy 1.26.4, and system NumPy 1.26.4.

- [ ] **Step 7: Commit metadata preparation**

```powershell
git add carr_operator.py tests/blender/test_carr_metadata_export.py
git commit -m "feat: add CArr NeuS metadata export"
```

---

### Task 6: Modal Frame-Major RGBA Rendering

**Files:**
- Modify: `carr_operator.py`
- Modify: `__init__.py:2, 98-107`
- Modify: `tests/blender/test_carr_metadata_export.py`
- Create: `tests/blender/test_carr_render.py`
- Create: `tests/blender/manual_carr_modal_setup.py`

**Interfaces:**
- Consumes: `PreparedExport` and `RenderTask` from Task 5.
- Produces: `configure_rgba(scene)`, `apply_calibration(scene, camera, calibration)`, `render_task(context, task, calibration)`, and registered `CameraArray` with `execute`, `modal`, `_start_queue`, and `_finish`.

- [ ] **Step 1: Write a failing render-order and direct-render integration script**

```python
from pathlib import Path
import tempfile

import bpy

from carr_test_support import registered_addon


with registered_addon() as addon, tempfile.TemporaryDirectory() as temporary:
    from BlenderNeRF import carr_operator
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.resolution_x = 8
    scene.render.resolution_y = 8
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.use_file_extension = False
    scene.render.film_transparent = True
    scene.frame_start = 1
    scene.frame_end = 2
    scene.save_path = temporary
    scene.carr_dataset_name = 'rendered'
    scene.carr_camera_count = 2
    scene.train_data = True
    scene.test_data = True
    scene.render_frames = True
    scene.logs = False

    prepared = carr_operator.prepare_export(bpy.context)
    assert [(task.blender_frame, task.split, task.camera.name) for task in prepared.tasks] == [
        (1, 'train', 'BlenderNeRF CArr Train 000'),
        (1, 'train', 'BlenderNeRF CArr Train 001'),
        (1, 'test', 'BlenderNeRF CArr Test'),
        (2, 'train', 'BlenderNeRF CArr Train 000'),
        (2, 'train', 'BlenderNeRF CArr Train 001'),
        (2, 'test', 'BlenderNeRF CArr Test'),
    ]
    carr_operator.configure_rgba(scene)
    for task in prepared.tasks:
        assert 'FINISHED' in carr_operator.render_task(bpy.context, task, prepared.calibration)
    carr_operator.restore_scene(scene, prepared.initial_state)

    pngs = sorted((Path(temporary) / 'rendered').glob('*/rgb/*.png'))
    assert len(pngs) == 6
    for png in pngs:
        data = png.read_bytes()
        assert data[:8] == b'\x89PNG\r\n\x1a\n'
        assert data[25] == 6
    assert scene.render.image_settings.file_format == 'JPEG'
    assert scene.render.image_settings.color_mode == 'RGB'
    assert scene.render.use_file_extension is False

print('CArr render checks passed')
```

- [ ] **Step 2: Run the render script and verify missing APIs**

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --background --factory-startup --python-exit-code 1 --python 'tests/blender/test_carr_render.py'
```

Expected: non-zero exit because `configure_rgba` and `render_task` are absent.

- [ ] **Step 3: Implement isolated RGBA rendering**

```python
def configure_rgba(scene):
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.use_file_extension = True


def apply_calibration(scene, camera, calibration):
    camera.data.lens = calibration.lens
    camera.data.sensor_width = calibration.sensor_width
    camera.data.sensor_height = calibration.sensor_height
    camera.data.sensor_fit = calibration.sensor_fit
    scene.render.resolution_x = calibration.resolution_x
    scene.render.resolution_y = calibration.resolution_y
    scene.render.resolution_percentage = calibration.resolution_percentage
    scene.render.pixel_aspect_x = calibration.pixel_aspect_x
    scene.render.pixel_aspect_y = calibration.pixel_aspect_y


def render_task(context, task, calibration):
    scene = context.scene
    scene.frame_set(task.blender_frame)
    apply_calibration(scene, task.camera, calibration)
    task.camera.matrix_world = task.matrix_world.copy()
    context.view_layer.update()
    scene.camera = task.camera
    scene.render.filepath = task.filepath
    return bpy.ops.render.render('EXEC_DEFAULT', write_still=True)
```

Ensure every Task 5 filepath is `<camera directory>/rgb/<three-digit stem>` without `.png`; Blender appends it because extensions are enabled.

- [ ] **Step 4: Run the direct render integration script**

Run the Step 2 command.

Expected: exit 0, six images, PNG IHDR color type 6, and restored JPEG/RGB settings.

- [ ] **Step 5: Implement the modal operator using the tested primitives**

```python
class CameraArray(bpy.types.Operator):
    bl_idname = 'object.camera_array'
    bl_label = 'Camera Array CArr'
    _is_running = False

    @classmethod
    def poll(cls, context):
        return not cls._is_running

    def execute(self, context):
        try:
            prepared = prepare_export(context)
        except CArrValidationError as exception:
            self.report({'ERROR'}, str(exception))
            return {'CANCELLED'}
        except Exception as exception:
            self.report({'ERROR'}, 'CArr export failed: {}'.format(exception))
            return {'CANCELLED'}
        if not context.scene.render_frames:
            complete_metadata_only(context, prepared)
            self.report({'INFO'}, 'CArr dataset saved to {}.'.format(prepared.output_path))
            return {'FINISHED'}
        return self._start_queue(context, prepared)
```

Implement `_start_queue` with the COS timer pattern: store the prepared export, set render index 0 and finalized false, set `_is_running`, configure RGBA, begin progress with the exact task count, add a 0.1-second event timer using `context.window`, register the modal handler, and return `RUNNING_MODAL`.

Implement `modal` with exact branches and call `render_task(context, task, self._prepared.calibration)` for the current task:

- `ESC`: finish cancelled between images.
- Non-`TIMER`: return `PASS_THROUGH`.
- Exhausted queue: finish successfully.
- Exception or Blender `CANCELLED`: report task camera name and Blender frame, then finish failed.
- Successful image: increment index, update progress, redraw screen areas, and continue or finish.

Implement idempotent `_finish` so all paths remove the timer, end progress, reset `_is_running`, restore Task 5 state, retain rig/output, report complete versus partial status, and return `FINISHED` or `CANCELLED`.

- [ ] **Step 6: Register the operator and test the metadata operator path**

Import `carr_operator` in `__init__.py` and append `carr_operator.CameraArray` to `CLASSES`. Add this assertion to `test_carr_metadata_export.py` using the same temporary directory:

```python
scene.carr_dataset_name = 'operator_metadata'
scene.render_frames = False
assert bpy.ops.object.camera_array() == {'FINISHED'}
assert (Path(temporary) / 'operator_metadata' / 'cam_test' / 'cameras_sphere.npz').is_file()
```

- [ ] **Step 7: Run automated render and operator checks on Blender 5.1 and 4.4**

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --background --factory-startup --python-exit-code 1 --python 'tests/blender/test_carr_metadata_export.py'
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --background --factory-startup --python-exit-code 1 --python 'tests/blender/test_carr_render.py'
& 'C:\Program Files\Blender Foundation\Blender 4.4\blender.exe' --background --factory-startup --python-exit-code 1 --python 'tests/blender/test_carr_render.py'
```

Expected: all three commands exit 0.

- [ ] **Step 8: Add and perform the interactive Blender 5.1 modal smoke**

Create `manual_carr_modal_setup.py` with executable setup content:

```python
from datetime import datetime
from pathlib import Path
import sys

import bpy


repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root.parent))
import BlenderNeRF

BlenderNeRF.register()
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE_NEXT'
scene.render.resolution_x = 64
scene.render.resolution_y = 64
scene.render.resolution_percentage = 100
scene.frame_start = 1
scene.frame_end = 5
scene.carr_geometry = 'CIRCLE'
scene.carr_camera_count = 3
scene.carr_radius = 4.0
scene.train_data = True
scene.test_data = True
scene.render_frames = True
scene.save_path = bpy.app.tempdir
scene.carr_dataset_name = 'carr_modal_manual_' + datetime.now().strftime('%Y%m%d_%H%M%S')
scene.carr_show_rig = True

bpy.ops.object.light_add(type='AREA', location=(2.0, -2.0, 4.0))
bpy.context.object.data.energy = 1000.0
output = Path(scene.save_path) / bpy.path.clean_name(scene.carr_dataset_name)
print('Open the BlenderNeRF N-panel, click PLAY CArr, then press Escape between images.')
print('Expected partial output:', output)
```

Run:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --factory-startup --python 'tests/blender/manual_carr_modal_setup.py'
```

Expected manual observations: Blender remains responsive between renders; progress advances through 20 tasks; Escape stops the queue; original frame/camera/filepath/format return; partial output remains. Close Blender without saving.

- [ ] **Step 9: Commit modal rendering**

```powershell
git add carr_operator.py __init__.py tests/blender/test_carr_metadata_export.py tests/blender/test_carr_render.py tests/blender/manual_carr_modal_setup.py
git commit -m "feat: render synchronized CArr datasets"
```

---

### Task 7: Panel, Documentation, Versioning, and Package Verification

**Files:**
- Create: `carr_ui.py`
- Create: `tests/blender/test_carr_registration.py`
- Modify: `__init__.py:2, 5-12, 98-107`
- Modify: `README.md:42-139`
- Modify: `blender_manifest.toml:3, 18-26`

**Interfaces:**
- Consumes: registered CArr properties and `object.camera_array` from prior tasks.
- Produces: `CARR_UI`, complete user documentation, synchronized version `6.2.0`, and a package excluding docs/tests.

- [ ] **Step 1: Write a failing final registration smoke script**

```python
import tomllib

import bpy

from carr_test_support import REPO_ROOT, registered_addon


with registered_addon() as addon:
    scene = bpy.context.scene
    assert hasattr(bpy.types, 'CARR_UI')
    assert hasattr(bpy.types, 'CameraArray')
    assert scene.carr_geometry == 'CIRCLE'
    assert scene.carr_camera_count == 10
    assert scene.carr_radius == 4.0
    assert scene.carr_focal == 50.0
    assert addon.bl_info['version'] == (6, 2, 0)
    manifest = tomllib.loads((REPO_ROOT / 'blender_manifest.toml').read_text(encoding='utf-8'))
    assert manifest['version'] == '6.2.0'
    assert manifest['blender_version_min'] == '4.2.0'
    assert addon.carr_rig.carr_frame_change in bpy.app.handlers.frame_change_post

print('CArr final registration checks passed')
```

- [ ] **Step 2: Run the registration smoke and verify UI/version failures**

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --background --factory-startup --python-exit-code 1 --python 'tests/blender/test_carr_registration.py'
```

Expected: non-zero exit because `CARR_UI` is absent and the version remains 6.1.0.

- [ ] **Step 3: Implement the dedicated panel**

```python
import bpy


class CARR_UI(bpy.types.Panel):
    bl_idname = 'VIEW3D_PT_carr_ui'
    bl_label = 'Camera Array CArr'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'BlenderNeRF'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.use_property_split = True
        layout.prop(scene, 'carr_geometry')
        layout.prop(scene, 'carr_camera_count')
        layout.prop(scene, 'carr_location')
        layout.prop(scene, 'carr_rotation')
        layout.prop(scene, 'carr_radius')
        layout.prop(scene, 'carr_focal')
        layout.use_property_split = False
        layout.separator()
        layout.label(text='Splits')
        row = layout.row(align=True)
        row.prop(scene, 'train_data', toggle=True)
        row.prop(scene, 'test_data', toggle=True)
        layout.separator()
        layout.label(text='Preview')
        layout.prop(scene, 'carr_show_rig', toggle=True)
        layout.separator()
        layout.use_property_split = True
        layout.prop(scene, 'carr_dataset_name')
        layout.separator()
        layout.operator('object.camera_array', text='PLAY CArr')
```

Import `carr_ui` and place `carr_ui.CARR_UI` with other panels before operators in `CLASSES`.

- [ ] **Step 4: Version and package exclusions**

Set `bl_info['version']` to `(6, 2, 0)`, set manifest `version = "6.2.0"`, add `"/tests/",` to `paths_exclude_pattern`, retain `/docs/`, and retain `blender_version_min = "4.2.0"`.

- [ ] **Step 5: Update README with the complete user contract**

Make these explicit changes:

1. Describe four methods and distinguish CArr directory output from the other methods' ZIP output.
2. Add CArr overview and `How to CArr` sections covering every CArr panel property.
3. State fixed train cameras, local +Z Hemisphere, shared intrinsics, and geometry-specific test paths.
4. Show the per-camera `rgb` plus `cameras_sphere.npz` tree.
5. State RGBA output, no masks/depth, identity scales, index synchronization, metadata-only behavior, and no ZIP.
6. State AABB, Gaussian Points, File Format, and Path Format do not affect CArr.
7. Document non-overwrite validation, modal progress, cancellation, restoration, and partial-output retention.

- [ ] **Step 6: Run all automated tests on primary Blender 5.1**

```powershell
python -m pytest tests/test_carr_geometry.py tests/test_carr_neus.py -v
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --background --factory-startup --python-exit-code 1 --python 'tests/blender/test_carr_preview.py'
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --background --factory-startup --python-exit-code 1 --python 'tests/blender/test_carr_metadata_export.py'
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --background --factory-startup --python-exit-code 1 --python 'tests/blender/test_carr_render.py'
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --background --factory-startup --python-exit-code 1 --python 'tests/blender/test_carr_registration.py'
```

Expected: pytest and all Blender scripts PASS.

- [ ] **Step 7: Run the compatibility smoke suite on Blender 4.4**

```powershell
& 'C:\Program Files\Blender Foundation\Blender 4.4\blender.exe' --background --factory-startup --python-exit-code 1 --python 'tests/blender/test_carr_preview.py'
& 'C:\Program Files\Blender Foundation\Blender 4.4\blender.exe' --background --factory-startup --python-exit-code 1 --python 'tests/blender/test_carr_metadata_export.py'
& 'C:\Program Files\Blender Foundation\Blender 4.4\blender.exe' --background --factory-startup --python-exit-code 1 --python 'tests/blender/test_carr_render.py'
& 'C:\Program Files\Blender Foundation\Blender 4.4\blender.exe' --background --factory-startup --python-exit-code 1 --python 'tests/blender/test_carr_registration.py'
```

Expected: all scripts exit 0 without post-4.2-only APIs.

- [ ] **Step 8: Validate and build the Blender 5.1 extension package**

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --command extension validate --source-dir .
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --command extension build --source-dir . --output-dir dist
tar -tf 'dist\BlenderNeRF-6.2.0.zip'
```

Expected: validation succeeds; the ZIP contains runtime Python and the manifest, with neither `docs/` nor `tests/`.

- [ ] **Step 9: Inspect scope and commit the completed feature**

```powershell
git status --short --untracked-files=all
git diff --check
git diff --stat
git add __init__.py carr_ui.py README.md blender_manifest.toml tests/blender/test_carr_registration.py
git commit -m "docs: expose and document Camera Array"
```

Expected: only the scoped CArr files are present and the commit succeeds.

- [ ] **Step 10: Run final post-commit verification**

```powershell
python -m pytest tests/test_carr_geometry.py tests/test_carr_neus.py -v
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --background --factory-startup --python-exit-code 1 --python 'tests/blender/test_carr_preview.py'
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --background --factory-startup --python-exit-code 1 --python 'tests/blender/test_carr_metadata_export.py'
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --background --factory-startup --python-exit-code 1 --python 'tests/blender/test_carr_render.py'
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --background --factory-startup --python-exit-code 1 --python 'tests/blender/test_carr_registration.py'
git status --short --untracked-files=all
```

Expected: every test passes and the worktree is clean.
