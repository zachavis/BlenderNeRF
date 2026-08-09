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
    expected = np.array([[2000.0, 0.0, 200.0], [0.0, 2000.0, 400.0], [0.0, 0.0, 1.0]])
    assert vertical == pytest.approx(expected)
    assert automatic == pytest.approx(expected)


def test_auto_intrinsics_use_pixel_aspect_for_effective_dimensions():
    actual = carr_neus.intrinsic_matrix(
        50.0, 36.0, 24.0, 'AUTO', 800, 600, 100.0, 1.0, 2.0
    )
    expected = np.array([
        [2500.0, 0.0, 400.0],
        [0.0, 1250.0, 300.0],
        [0.0, 0.0, 1.0],
    ])
    assert actual == pytest.approx(expected)


def test_intrinsics_apply_render_resolution_percentage():
    actual = carr_neus.intrinsic_matrix(
        50.0, 10.0, 10.0, 'HORIZONTAL', 800, 600, 25.0, 1.0, 1.0
    )
    expected = np.array([
        [1000.0, 0.0, 100.0],
        [0.0, 1000.0, 75.0],
        [0.0, 0.0, 1.0],
    ])
    assert actual == pytest.approx(expected)


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
    assert loaded['world_mat_2'].shape == (3, 4)
    assert loaded['world_mat_2'].dtype == np.float64
    assert loaded['scale_mat_2'].shape == (4, 4)
    assert loaded['scale_mat_2'].dtype == np.float32
    assert loaded['scale_mat_2'] == pytest.approx(np.eye(4, dtype=np.float32))
