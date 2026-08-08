import math


CIRCLE = 'CIRCLE'
HEMISPHERE = 'HEMISPHERE'
SPHERE = 'SPHERE'
GEOMETRIES = (CIRCLE, HEMISPHERE, SPHERE)
GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


def _surface_point(z, theta):
    rho = math.sqrt(max(0.0, 1.0 - z * z))
    return (rho * math.cos(theta), rho * math.sin(theta), z)


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
