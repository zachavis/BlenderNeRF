import math

import pytest

import carr_geometry


def length(vector):
    return math.sqrt(sum(component * component for component in vector))


def test_circle_has_exact_count_unit_radius_and_equal_angles():
    points = carr_geometry.train_local_positions(carr_geometry.CIRCLE, 4)
    expected = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
    ]
    assert len(points) == len(expected)
    for point, expected_point in zip(points, expected):
        assert point == pytest.approx(expected_point)


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
