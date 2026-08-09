from pathlib import Path
import sys
import tempfile

import bpy
import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from carr_test_support import registered_addon


with registered_addon() as addon, tempfile.TemporaryDirectory() as temporary:
    from BlenderNeRF import carr_operator
    scene = bpy.context.scene
    scene.frame_start = 10
    scene.frame_end = 12
    scene.frame_set(11)
    original = (scene.frame_current, scene.camera, scene.render.filepath)
    original_render = (
        scene.render.image_settings.file_format,
        scene.render.image_settings.color_mode,
        scene.render.use_file_extension,
        scene.render.resolution_x,
        scene.render.resolution_y,
        scene.render.resolution_percentage,
        scene.render.pixel_aspect_x,
        scene.render.pixel_aspect_y,
    )
    scene.save_path = temporary
    scene.carr_dataset_name = 'synced array'
    scene.carr_geometry = 'CIRCLE'
    scene.carr_camera_count = 2
    scene.train_data = True
    scene.test_data = True
    scene.render_frames = False
    scene.logs = True

    prepared = carr_operator.prepare_export(bpy.context)
    assert prepared.frames == [10, 11, 12]
    assert [task.split for task in prepared.tasks] == [
        'train', 'train', 'test',
        'train', 'train', 'test',
        'train', 'train', 'test',
    ]
    assert [task.blender_frame for task in prepared.tasks] == [10, 10, 10, 11, 11, 11, 12, 12, 12]
    assert [task.output_index for task in prepared.tasks] == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert [Path(task.filepath).name for task in prepared.tasks] == [
        '000', '000', '000',
        '001', '001', '001',
        '002', '002', '002',
    ]
    original_camera_settings = (
        prepared.test_camera.data.lens,
        prepared.test_camera.data.sensor_width,
        prepared.test_camera.data.sensor_height,
        prepared.test_camera.data.sensor_fit,
    )
    scene.frame_set(12)
    scene.camera = prepared.test_camera
    scene.render.filepath = 'changed'
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.image_settings.color_mode = 'BW'
    scene.render.use_file_extension = False
    scene.render.resolution_x = 320
    scene.render.resolution_y = 240
    scene.render.resolution_percentage = 25
    scene.render.pixel_aspect_x = 2.0
    scene.render.pixel_aspect_y = 3.0
    prepared.test_camera.data.lens = 18.0
    prepared.test_camera.data.sensor_width = 20.0
    prepared.test_camera.data.sensor_height = 10.0
    prepared.test_camera.data.sensor_fit = 'VERTICAL'
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
    assert (
        scene.render.image_settings.file_format,
        scene.render.image_settings.color_mode,
        scene.render.use_file_extension,
        scene.render.resolution_x,
        scene.render.resolution_y,
        scene.render.resolution_percentage,
        scene.render.pixel_aspect_x,
        scene.render.pixel_aspect_y,
    ) == original_render
    assert (
        prepared.test_camera.data.lens,
        prepared.test_camera.data.sensor_width,
        prepared.test_camera.data.sensor_height,
        prepared.test_camera.data.sensor_fit,
    ) == original_camera_settings
    train.close()
    test.close()
    assert scene.save_path == temporary

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

    blocked_ancestor = Path(temporary) / 'blocked'
    blocked_ancestor.write_text('not a directory', encoding='utf-8')
    scene.save_path = str(blocked_ancestor / 'nested')
    scene.carr_dataset_name = 'unusable target'
    try:
        carr_operator.validate_export(scene)
    except carr_operator.CArrValidationError as exception:
        assert 'usable' in str(exception)
    else:
        raise AssertionError('CArr accepted a target beneath a file')

    scene.save_path = temporary
    scene.carr_dataset_name = 'empty unwritable'
    unwritable_target = Path(temporary) / 'empty_unwritable'
    unwritable_target.mkdir()
    original_access = carr_operator.os.access

    def target_is_not_writable(path, mode):
        if Path(path) == unwritable_target:
            return False
        return original_access(path, mode)

    carr_operator.os.access = target_is_not_writable
    try:
        try:
            carr_operator.validate_export(scene)
        except carr_operator.CArrValidationError as exception:
            assert 'usable' in str(exception)
        else:
            raise AssertionError('CArr accepted an existing unwritable target')
    finally:
        carr_operator.os.access = original_access

    broken_link = Path(temporary) / 'broken_link'
    scene.save_path = str(broken_link)
    scene.carr_dataset_name = 'dataset'
    original_lexists = carr_operator.os.path.lexists

    def broken_link_lexists(path):
        if Path(path) == broken_link:
            return True
        return original_lexists(path)

    carr_operator.os.path.lexists = broken_link_lexists
    try:
        try:
            carr_operator.validate_export(scene)
        except carr_operator.CArrValidationError as exception:
            assert 'symlink' in str(exception).lower()
        else:
            raise AssertionError('CArr accepted a broken save-path symlink')
    finally:
        carr_operator.os.path.lexists = original_lexists

    scene.save_path = temporary
    scene.carr_dataset_name = 'operator_metadata'
    scene.render_frames = False
    assert bpy.ops.object.camera_array() == {'FINISHED'}
    assert (Path(temporary) / 'operator_metadata' / 'cam_test' / 'cameras_sphere.npz').is_file()

with registered_addon(), tempfile.TemporaryDirectory() as temporary:
    from BlenderNeRF import carr_operator, carr_rig
    scene = bpy.context.scene
    foreign_test = bpy.data.objects.new(carr_rig.CARR_TEST_NAME, None)
    scene.collection.objects.link(foreign_test)
    scene.frame_start = 1
    scene.frame_end = 2
    scene.frame_set(1)
    scene.carr_camera_count = 2
    scene.train_data = False
    scene.test_data = True
    scene.render_frames = False
    scene.logs = False
    scene.carr_dataset_name = 'reserved collision'
    scene.save_path = temporary

    prepared = carr_operator.prepare_export(bpy.context)
    assert prepared.test_camera is not foreign_test
    assert carr_operator.complete_metadata_only(bpy.context, prepared) == {'FINISHED'}
    assert foreign_test.name in bpy.data.objects

bpy.data.objects.remove(foreign_test, do_unlink=True)

with registered_addon(), tempfile.TemporaryDirectory() as temporary:
    from BlenderNeRF import carr_operator
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 2
    scene.frame_set(1)
    scene.carr_camera_count = 2
    scene.train_data = False
    scene.test_data = True
    scene.carr_dataset_name = 'preview failure'
    scene.save_path = temporary
    original_initial_state = carr_operator._initial_state

    def failing_initial_state(current_scene, camera):
        current_scene.save_path = ''
        raise RuntimeError('forced preview failure')

    carr_operator._initial_state = failing_initial_state
    try:
        try:
            carr_operator.prepare_export(bpy.context)
        except RuntimeError as exception:
            assert 'forced preview failure' in str(exception)
        else:
            raise AssertionError('CArr swallowed a preview preparation failure')
    finally:
        carr_operator._initial_state = original_initial_state
    assert scene.save_path == temporary

print('CArr metadata export checks passed')
