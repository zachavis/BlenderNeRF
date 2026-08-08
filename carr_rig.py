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


def _remove_managed_objects(collection):
    for obj in list(collection.objects):
        if obj.get(MANAGED_KEY):
            bpy.data.objects.remove(obj, do_unlink=True)


def _remove_unused_managed_cameras():
    for camera in list(bpy.data.cameras):
        if camera.get(MANAGED_KEY) and camera.users == 0:
            bpy.data.cameras.remove(camera)


def _managed_collections():
    return [collection for collection in bpy.data.collections if collection.get(MANAGED_KEY)]


def _managed_test_camera(scene):
    for collection in scene.collection.children:
        if not collection.get(MANAGED_KEY):
            continue
        for obj in collection.objects:
            if (
                obj.get(MANAGED_KEY)
                and obj.type == 'CAMERA'
                and obj.name.startswith(CARR_TEST_NAME)
            ):
                return obj
    return None


def ensure_preview(context):
    scene = context.scene
    collections = _managed_collections()
    if collections:
        collection = collections[0]
    else:
        collection = bpy.data.collections.new(CARR_COLLECTION_NAME)
        collection[MANAGED_KEY] = True
    if collection.name not in scene.collection.children:
        scene.collection.children.link(collection)

    _remove_managed_objects(collection)
    _remove_unused_managed_cameras()

    rig = bpy.data.objects.new(CARR_RIG_NAME, None)
    rig[MANAGED_KEY] = True
    collection.objects.link(rig)

    camera_data = bpy.data.cameras.new(CARR_CAMERA_DATA_NAME)
    camera_data.type = 'PERSP'
    camera_data.lens = scene.carr_focal
    camera_data[MANAGED_KEY] = True

    train_objects = []
    for index, matrix in enumerate(train_camera_matrices(scene)):
        camera = bpy.data.objects.new(CARR_TRAIN_PREFIX + '{:03d}'.format(index), camera_data)
        camera[MANAGED_KEY] = True
        collection.objects.link(camera)
        camera.matrix_world = matrix
        train_objects.append(camera)

    test = bpy.data.objects.new(CARR_TEST_NAME, camera_data)
    test[MANAGED_KEY] = True
    collection.objects.link(test)
    frame_count = scene.frame_end - scene.frame_start + 1
    output_index = scene.frame_current - scene.frame_start
    test.matrix_world = test_camera_matrix(scene, output_index, frame_count)
    context.view_layer.update()
    return train_objects, test


def remove_preview(scene):
    for collection in _managed_collections():
        _remove_managed_objects(collection)
        if not collection.objects:
            bpy.data.collections.remove(collection)
    _remove_unused_managed_cameras()


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
    if not scene.carr_show_rig:
        return
    test = _managed_test_camera(scene)
    if test is None:
        return
    frame_count = scene.frame_end - scene.frame_start + 1
    output_index = scene.frame_current - scene.frame_start
    test.matrix_world = test_camera_matrix(scene, output_index, frame_count)
