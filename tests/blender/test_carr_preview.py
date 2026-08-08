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
