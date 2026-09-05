import math
from pathlib import Path
import sys

import bpy

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from carr_test_support import matrix_rows, registered_addon


with registered_addon() as addon:
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 5
    scene.carr_geometry = 'CIRCLE'
    scene.carr_camera_count = 3
    scene.carr_location = (1.0, 2.0, 3.0)
    scene.carr_look_at = (1.0, 2.0, 3.0)
    scene.carr_rotation = (0.0, 0.0, math.pi / 2.0)
    scene.carr_radius = 4.0
    scene.carr_focal = 50.0
    scene.frame_set(-20)
    scene.carr_show_rig = True

    collection = bpy.data.collections[addon.carr_rig.CARR_COLLECTION_NAME]
    assert 'BlenderNeRF CArr Look-at' in collection.objects
    look_at = collection.objects['BlenderNeRF CArr Look-at']
    train = sorted(
        [obj for obj in collection.objects if obj.name.startswith(addon.carr_rig.CARR_TRAIN_PREFIX)],
        key=lambda obj: obj.name,
    )
    test = collection.objects[addon.carr_rig.CARR_TEST_NAME]
    assert len(train) == 3
    assert all(obj.type == 'CAMERA' for obj in train + [test])
    assert all(obj.data is train[0].data for obj in train + [test])
    assert look_at.type == 'EMPTY'
    assert look_at.empty_display_type == 'SPHERE'
    assert tuple(look_at.location) == (1.0, 2.0, 3.0)
    assert train[0].data.lens == 50.0
    assert all(
        math.isclose(
            sum((obj.location[index] - scene.carr_location[index]) ** 2 for index in range(3)),
            16.0,
            rel_tol=1.0e-6,
        )
        for obj in train
    )
    first_forward = tuple(-matrix_rows(train[0].matrix_world)[row][2] for row in range(3))
    assert all(
        math.isclose(actual, expected, abs_tol=1.0e-6)
        for actual, expected in zip(first_forward, (0.0, -1.0, 0.0))
    )

    scene.carr_look_at = (1.0, 2.0, 6.0)
    collection = bpy.data.collections[addon.carr_rig.CARR_COLLECTION_NAME]
    look_at = collection.objects['BlenderNeRF CArr Look-at']
    train = sorted(
        [obj for obj in collection.objects if obj.name.startswith(addon.carr_rig.CARR_TRAIN_PREFIX)],
        key=lambda obj: obj.name,
    )
    test = collection.objects[addon.carr_rig.CARR_TEST_NAME]
    assert tuple(look_at.location) == (1.0, 2.0, 6.0)
    first_forward = tuple(-matrix_rows(train[0].matrix_world)[row][2] for row in range(3))
    assert all(
        math.isclose(actual, expected, abs_tol=1.0e-6)
        for actual, expected in zip(first_forward, (0.0, -0.8, 0.6))
    )
    expected_clamped = matrix_rows(addon.carr_rig.test_camera_matrix(scene, 0, 5))
    assert all(
        math.isclose(actual, expected, abs_tol=1.0e-6)
        for actual_row, expected_row in zip(matrix_rows(test.matrix_world), expected_clamped)
        for actual, expected in zip(actual_row, expected_row)
    )

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
    scene.carr_look_at = (1.0, 2.0, 0.0)
    collection = bpy.data.collections[addon.carr_rig.CARR_COLLECTION_NAME]
    look_at = collection.objects['BlenderNeRF CArr Look-at']
    train = sorted(
        [obj for obj in collection.objects if obj.name.startswith(addon.carr_rig.CARR_TRAIN_PREFIX)],
        key=lambda obj: obj.name,
    )
    assert tuple(look_at.location) == (1.0, 2.0, 0.0)
    first_forward = tuple(-matrix_rows(train[0].matrix_world)[row][2] for row in range(3))
    assert all(
        math.isclose(actual, expected, abs_tol=1.0e-6)
        for actual, expected in zip(first_forward, (0.0, -0.8, -0.6))
    )

    collision_target = tuple(train[0].location)
    scene.carr_look_at = collision_target
    assert not [collection for collection in bpy.data.collections if collection.get(addon.carr_rig.MANAGED_KEY)]
    assert not [obj for obj in bpy.data.objects if obj.get(addon.carr_rig.MANAGED_KEY)]
    assert not [camera for camera in bpy.data.cameras if camera.get(addon.carr_rig.MANAGED_KEY)]
    scene.frame_set(4)

    scene.carr_look_at = (1.0, 2.0, 0.0)
    collection = bpy.data.collections[addon.carr_rig.CARR_COLLECTION_NAME]
    assert 'BlenderNeRF CArr Look-at' in collection.objects
    scene.carr_show_rig = False
    assert addon.carr_rig.CARR_COLLECTION_NAME not in bpy.data.collections


def test_reserved_name_collisions_preserve_user_data_and_animate_managed_camera():
    with registered_addon() as addon:
        scene = bpy.context.scene
        scene.frame_start = 1
        scene.frame_end = 5
        foreign_collection = bpy.data.collections.new(addon.carr_rig.CARR_COLLECTION_NAME)
        scene.collection.children.link(foreign_collection)
        foreign_test = bpy.data.objects.new(addon.carr_rig.CARR_TEST_NAME, None)
        foreign_test.location = (8.0, 9.0, 10.0)
        foreign_collection.objects.link(foreign_test)

        scene.carr_show_rig = True
        managed_collections = [
            collection for collection in bpy.data.collections
            if collection.get(addon.carr_rig.MANAGED_KEY)
        ]
        assert len(managed_collections) == 1
        managed_collection = managed_collections[0]
        assert managed_collection is not foreign_collection
        managed_test = next(
            obj for obj in managed_collection.objects
            if obj.get(addon.carr_rig.MANAGED_KEY)
            and obj.type == 'CAMERA'
            and obj.name.startswith(addon.carr_rig.CARR_TEST_NAME)
        )

        scene.frame_set(1)
        managed_start = matrix_rows(managed_test.matrix_world)
        scene.frame_set(3)
        managed_middle = matrix_rows(managed_test.matrix_world)
        assert tuple(foreign_test.location) == (8.0, 9.0, 10.0)
        assert managed_start != managed_middle

        scene.carr_show_rig = False
        assert foreign_collection.name in bpy.data.collections
        assert foreign_test.name in bpy.data.objects
        assert not [
            collection for collection in bpy.data.collections
            if collection.get(addon.carr_rig.MANAGED_KEY)
        ]

    bpy.data.objects.remove(foreign_test, do_unlink=True)
    bpy.data.collections.remove(foreign_collection)


test_reserved_name_collisions_preserve_user_data_and_animate_managed_camera()


def test_unregister_removes_visible_managed_preview():
    addon.register()
    scene = bpy.context.scene
    scene.carr_show_rig = True
    assert [collection for collection in bpy.data.collections if collection.get(addon.carr_rig.MANAGED_KEY)]
    assert [obj for obj in bpy.data.objects if obj.get(addon.carr_rig.MANAGED_KEY)]
    assert [camera for camera in bpy.data.cameras if camera.get(addon.carr_rig.MANAGED_KEY)]

    addon.unregister()

    assert not [collection for collection in bpy.data.collections if collection.get(addon.carr_rig.MANAGED_KEY)]
    assert not [obj for obj in bpy.data.objects if obj.get(addon.carr_rig.MANAGED_KEY)]
    assert not [camera for camera in bpy.data.cameras if camera.get(addon.carr_rig.MANAGED_KEY)]


test_unregister_removes_visible_managed_preview()

print('CArr preview checks passed')
