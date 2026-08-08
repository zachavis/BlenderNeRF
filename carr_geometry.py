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
