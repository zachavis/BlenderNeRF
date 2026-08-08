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
        fit = 'VERTICAL' if aspect_x * width <= aspect_y * height else 'HORIZONTAL'
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
