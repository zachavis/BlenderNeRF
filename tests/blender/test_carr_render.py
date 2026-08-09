from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

import bpy


TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from carr_test_support import registered_addon


class ModalHarness:
    _is_running = False
    modal = None
    _finish = None

    def __init__(self, prepared):
        self._prepared = prepared
        self._render_index = 0
        self._finalized = False
        self._timer = None
        self._progress_started = True
        self.reports = []

    def report(self, report_type, message):
        self.reports.append((report_type, message))


class FailingCleanupWindowManager:
    def __init__(self, wrapped, fail_timer=False, fail_progress=False):
        self.wrapped = wrapped
        self.fail_timer = fail_timer
        self.fail_progress = fail_progress
        self.timer_remove_attempts = 0
        self.progress_end_attempts = 0

    def event_timer_remove(self, timer):
        self.timer_remove_attempts += 1
        if self.fail_timer:
            raise RuntimeError('forced timer cleanup failure')
        self.wrapped.event_timer_remove(timer)

    def progress_end(self):
        self.progress_end_attempts += 1
        if self.fail_progress:
            raise RuntimeError('forced progress cleanup failure')
        self.wrapped.progress_end()


class FailingCameraData:
    def __setattr__(self, name, value):
        raise RuntimeError('forced camera restoration failure')


with registered_addon() as addon, tempfile.TemporaryDirectory() as temporary:
    from BlenderNeRF import carr_operator
    scene = bpy.context.scene
    engine_items = scene.render.bl_rna.properties['engine'].enum_items.keys()
    scene.render.engine = (
        'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in engine_items else 'BLENDER_EEVEE'
    )
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

    ModalHarness.modal = carr_operator.CameraArray.modal
    ModalHarness._finish = carr_operator.CameraArray._finish

    def start_modal(dataset_name, train_data, test_data):
        scene.carr_dataset_name = dataset_name
        scene.train_data = train_data
        scene.test_data = test_data
        scene.frame_start = 3
        scene.frame_end = 3
        scene.frame_set(3)
        current = carr_operator.prepare_export(bpy.context)
        carr_operator.configure_rgba(scene)
        bpy.context.window_manager.progress_begin(0, len(current.tasks))
        ModalHarness._is_running = True
        return current

    def assert_restored(current):
        state = current.initial_state
        assert scene.frame_current == state['frame']
        assert scene.camera == state['camera']
        assert scene.render.filepath == state['render_filepath']
        assert scene.render.image_settings.file_format == state['file_format']
        assert scene.render.image_settings.color_mode == state['color_mode']
        assert scene.render.use_file_extension == state['use_file_extension']
        assert scene.render.resolution_x == state['resolution_x']
        assert scene.render.resolution_y == state['resolution_y']
        assert scene.render.resolution_percentage == state['resolution_percentage']
        assert scene.render.pixel_aspect_x == state['pixel_aspect_x']
        assert scene.render.pixel_aspect_y == state['pixel_aspect_y']
        assert scene.render.film_transparent is True
        assert current.test_camera.data.lens == state['lens']
        assert current.test_camera.data.sensor_width == state['sensor_width']
        assert current.test_camera.data.sensor_height == state['sensor_height']
        assert current.test_camera.data.sensor_fit == state['sensor_fit']

    timer_event = SimpleNamespace(type='TIMER')
    escape_event = SimpleNamespace(type='ESC')

    successful = start_modal('modal_success', False, True)
    successful_harness = ModalHarness(successful)
    assert successful_harness.modal(bpy.context, timer_event) == {'FINISHED'}
    assert len(list((Path(successful.output_path) / 'cam_test' / 'rgb').glob('*.png'))) == 1
    assert_restored(successful)

    cancelled = start_modal('modal_cancelled', True, False)
    cancelled_harness = ModalHarness(cancelled)
    assert cancelled_harness.modal(bpy.context, timer_event) == {'RUNNING_MODAL'}
    assert cancelled_harness.modal(bpy.context, escape_event) == {'CANCELLED'}
    assert len(list(Path(cancelled.output_path).glob('*/rgb/*.png'))) == 1
    assert 'Partial files remain' in cancelled_harness.reports[-1][1]
    assert_restored(cancelled)

    failed = start_modal('modal_failed', True, False)
    failed_harness = ModalHarness(failed)
    assert failed_harness.modal(bpy.context, timer_event) == {'RUNNING_MODAL'}
    failed.tasks[1] = replace(failed.tasks[1], matrix_world=None)
    assert failed_harness.modal(bpy.context, timer_event) == {'CANCELLED'}
    assert len(list(Path(failed.output_path).glob('*/rgb/*.png'))) == 1
    assert failed.tasks[1].camera.name in failed_harness.reports[0][1]
    assert 'Blender frame 3' in failed_harness.reports[0][1]
    assert 'Partial files remain' in failed_harness.reports[-1][1]
    assert_restored(failed)

    live_edit = start_modal('modal_live_edit', True, False)
    live_edit_harness = ModalHarness(live_edit)
    carr_operator.CameraArray._is_running = True
    assert live_edit_harness.modal(bpy.context, timer_event) == {'RUNNING_MODAL'}
    queued_camera = live_edit.tasks[1].camera
    scene.carr_camera_count = 3
    scene.carr_show_rig = False
    assert queued_camera.name in bpy.data.objects
    carr_operator.CameraArray._is_running = False
    assert live_edit_harness.modal(bpy.context, timer_event) == {'FINISHED'}
    assert len(list(Path(live_edit.output_path).glob('*/rgb/*.png'))) == 2
    assert live_edit_harness._finalized is True

    deleted_camera = start_modal('modal_deleted_camera', True, False)
    deleted_camera_harness = ModalHarness(deleted_camera)
    task = deleted_camera.tasks[0]
    expected_label = task.camera_label
    bpy.data.objects.remove(task.camera, do_unlink=True)
    assert deleted_camera_harness.modal(bpy.context, timer_event) == {'CANCELLED'}
    assert expected_label in deleted_camera_harness.reports[0][1]
    assert deleted_camera_harness._finalized is True
    assert ModalHarness._is_running is False

    restoration_failure = start_modal('restoration_failure', False, True)
    restoration_failure_harness = ModalHarness(restoration_failure)
    original_frame = restoration_failure.initial_state['frame']
    original_camera = restoration_failure.initial_state['camera']
    restoration_failure.initial_state['camera_data'] = FailingCameraData()
    scene.frame_set(original_frame + 5)
    scene.camera = restoration_failure.test_camera
    assert restoration_failure_harness._finish(bpy.context, True) == {'CANCELLED'}
    assert scene.frame_current == original_frame
    assert scene.camera == original_camera
    assert restoration_failure_harness.reports[-1][0] == {'ERROR'}
    assert 'restoration failed' in restoration_failure_harness.reports[-1][1].lower()

    timer_cleanup = start_modal('timer_cleanup_failure', False, True)
    timer_cleanup_harness = ModalHarness(timer_cleanup)
    timer_cleanup_harness._timer = object()
    timer_cleanup_manager = FailingCleanupWindowManager(
        bpy.context.window_manager, fail_timer=True
    )
    timer_cleanup_context = SimpleNamespace(
        scene=scene, window_manager=timer_cleanup_manager
    )
    assert timer_cleanup_harness._finish(timer_cleanup_context, True) == {'FINISHED'}
    assert timer_cleanup_manager.timer_remove_attempts == 1
    assert timer_cleanup_manager.progress_end_attempts == 1
    assert_restored(timer_cleanup)
    assert timer_cleanup_harness.reports[-1][0] == {'INFO'}
    assert timer_cleanup_harness._finish(timer_cleanup_context, True) == {'FINISHED'}
    assert timer_cleanup_manager.timer_remove_attempts == 1
    assert len(timer_cleanup_harness.reports) == 1

    progress_cleanup = start_modal('progress_cleanup_failure', False, True)
    progress_cleanup_harness = ModalHarness(progress_cleanup)
    progress_cleanup_manager = FailingCleanupWindowManager(
        bpy.context.window_manager, fail_progress=True
    )
    progress_cleanup_context = SimpleNamespace(
        scene=scene, window_manager=progress_cleanup_manager
    )
    assert progress_cleanup_harness._finish(progress_cleanup_context, False) == {'CANCELLED'}
    assert progress_cleanup_manager.progress_end_attempts == 1
    assert_restored(progress_cleanup)
    assert progress_cleanup_harness.reports[-1][0] == {'WARNING'}
    assert 'Partial files remain' in progress_cleanup_harness.reports[-1][1]
    assert progress_cleanup_harness._finish(progress_cleanup_context, False) == {'CANCELLED'}
    assert progress_cleanup_manager.progress_end_attempts == 1
    assert len(progress_cleanup_harness.reports) == 1

print('CArr render checks passed')
