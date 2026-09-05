from dataclasses import dataclass
import datetime
import json
import os
from pathlib import Path

import bpy

from . import carr_neus, carr_rig


class CArrValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class RenderCalibration:
    lens: float
    sensor_width: float
    sensor_height: float
    sensor_fit: str
    resolution_x: int
    resolution_y: int
    resolution_percentage: int
    pixel_aspect_x: float
    pixel_aspect_y: float


@dataclass(frozen=True)
class RenderTask:
    split: str
    camera: object
    camera_label: str
    blender_frame: int
    output_index: int
    filepath: str
    matrix_world: object


@dataclass
class PreparedExport:
    output_path: str
    frames: list
    tasks: list
    initial_state: dict
    train_cameras: list
    test_camera: object
    calibration: RenderCalibration


def validate_export(scene):
    if not (scene.train_data or scene.test_data):
        raise CArrValidationError('CArr requires at least one train or test split')
    if scene.carr_camera_count < 2:
        raise CArrValidationError('CArr camera count must be at least 2')
    if scene.carr_radius <= 0:
        raise CArrValidationError('CArr radius must be positive')
    if not scene.save_path:
        raise CArrValidationError('CArr save path cannot be empty')

    dataset_name = bpy.path.clean_name(scene.carr_dataset_name)
    if not dataset_name:
        raise CArrValidationError('CArr dataset name cannot be empty after sanitizing')
    if not carr_neus.numpy_available():
        raise CArrValidationError('CArr NeuS export requires NumPy in Blender Python')
    if scene.frame_end < scene.frame_start:
        raise CArrValidationError('CArr frame range cannot be reversed')

    save_path = Path(bpy.path.abspath(scene.save_path))
    output_path = save_path / dataset_name
    try:
        existing_target = None
        for component in list(reversed(output_path.parents)) + [output_path]:
            if not os.path.lexists(component):
                continue
            if not component.exists():
                raise CArrValidationError('CArr path contains a broken symlink: {}'.format(component))
            if not component.is_dir():
                raise CArrValidationError('CArr path component is not a usable directory: {}'.format(component))
            existing_target = component
        if existing_target is None or not os.access(existing_target, os.W_OK):
            raise CArrValidationError('CArr target path is not a usable directory: {}'.format(output_path))
        if output_path.is_dir() and any(output_path.iterdir()):
            raise CArrValidationError('CArr refuses to use an existing non-empty target: {}'.format(output_path))
    except OSError as exception:
        raise CArrValidationError('CArr target path is not usable: {}'.format(output_path)) from exception

    frames = list(range(scene.frame_start, scene.frame_end + 1))
    try:
        carr_rig.train_camera_matrices(scene)
        for output_index in range(len(frames)):
            carr_rig.test_camera_matrix(scene, output_index, len(frames))
    except ValueError as exception:
        raise CArrValidationError(str(exception)) from exception
    return str(output_path), frames


def _calibration(scene, camera):
    data = camera.data
    render = scene.render
    return RenderCalibration(
        lens=float(data.lens),
        sensor_width=float(data.sensor_width),
        sensor_height=float(data.sensor_height),
        sensor_fit=str(data.sensor_fit),
        resolution_x=int(render.resolution_x),
        resolution_y=int(render.resolution_y),
        resolution_percentage=int(render.resolution_percentage),
        pixel_aspect_x=float(render.pixel_aspect_x),
        pixel_aspect_y=float(render.pixel_aspect_y),
    )


def _intrinsics(calibration):
    return carr_neus.intrinsic_matrix(
        calibration.lens,
        calibration.sensor_width,
        calibration.sensor_height,
        calibration.sensor_fit,
        calibration.resolution_x,
        calibration.resolution_y,
        calibration.resolution_percentage,
        calibration.pixel_aspect_x,
        calibration.pixel_aspect_y,
    )


def _projection(matrix_world, intrinsics):
    rows = tuple(tuple(float(value) for value in row) for row in matrix_world)
    return carr_neus.world_projection(rows, intrinsics)


def _initial_state(scene, camera):
    calibration = _calibration(scene, camera)
    return {
        'frame': scene.frame_current,
        'camera': scene.camera,
        'render_filepath': scene.render.filepath,
        'file_format': scene.render.image_settings.file_format,
        'color_mode': scene.render.image_settings.color_mode,
        'use_file_extension': scene.render.use_file_extension,
        'camera_data': camera.data,
        'lens': calibration.lens,
        'sensor_width': calibration.sensor_width,
        'sensor_height': calibration.sensor_height,
        'sensor_fit': calibration.sensor_fit,
        'resolution_x': calibration.resolution_x,
        'resolution_y': calibration.resolution_y,
        'resolution_percentage': calibration.resolution_percentage,
        'pixel_aspect_x': calibration.pixel_aspect_x,
        'pixel_aspect_y': calibration.pixel_aspect_y,
    }


def _camera_root(output_path, split, index=None):
    name = 'cam_test' if split == 'test' else 'cam_train_{}'.format(index)
    return Path(output_path) / name


def _render_filepath(camera_root, output_index):
    filename = carr_neus.image_filename(output_index)
    return str(camera_root / 'rgb' / os.path.splitext(filename)[0])


def _write_log(scene, output_path, frame_count):
    logdata = {
        'BlenderNeRF Version': scene.blendernerf_version,
        'Date and Time': datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
        'Train': scene.train_data,
        'Test': scene.test_data,
        'Render Frames': scene.render_frames,
        'Save Path': scene.save_path,
        'Method': 'CArr',
        'Geometry': scene.carr_geometry,
        'Camera Count': scene.carr_camera_count,
        'Location': str(list(scene.carr_location)),
        'Look-at': str(list(scene.carr_look_at)),
        'Rotation': str(list(scene.carr_rotation)),
        'Radius': scene.carr_radius,
        'Lens': str(scene.carr_focal) + ' mm',
        'Frame Count': frame_count,
        'Dataset Name': scene.carr_dataset_name,
    }
    with (Path(output_path) / 'log.txt').open('w', encoding='utf-8') as log_file:
        json.dump(logdata, log_file, indent=4)


def prepare_export(context):
    scene = context.scene
    requested_save_path = scene.save_path
    output_path, frames = validate_export(scene)
    initial_state = None
    try:
        scene.carr_show_rig = True
        train_cameras, test_camera = carr_rig.ensure_preview(context)
        scene.save_path = requested_save_path
        initial_state = _initial_state(scene, test_camera)
        calibration = _calibration(scene, test_camera)

        intrinsics = _intrinsics(calibration)
        train_matrices = [camera.matrix_world.copy() for camera in train_cameras]
        test_matrices = [
            carr_rig.test_camera_matrix(scene, output_index, len(frames)).copy()
            for output_index in range(len(frames))
        ]

        selected_train = list(enumerate(train_cameras)) if scene.train_data else []
        if scene.train_data:
            for camera_index, matrix_world in enumerate(train_matrices):
                camera_root = _camera_root(output_path, 'train', camera_index)
                (camera_root / 'rgb').mkdir(parents=True, exist_ok=True)
                projection = _projection(matrix_world, intrinsics)
                carr_neus.write_camera_archive(
                    camera_root / 'cameras_sphere.npz',
                    [projection for _ in frames],
                )
        if scene.test_data:
            camera_root = _camera_root(output_path, 'test')
            (camera_root / 'rgb').mkdir(parents=True, exist_ok=True)
            carr_neus.write_camera_archive(
                camera_root / 'cameras_sphere.npz',
                [_projection(matrix_world, intrinsics) for matrix_world in test_matrices],
            )

        tasks = []
        for output_index, blender_frame in enumerate(frames):
            for camera_index, camera in selected_train:
                tasks.append(RenderTask(
                    split='train',
                    camera=camera,
                    camera_label=camera.name,
                    blender_frame=blender_frame,
                    output_index=output_index,
                    filepath=_render_filepath(_camera_root(output_path, 'train', camera_index), output_index),
                    matrix_world=train_matrices[camera_index].copy(),
                ))
            if scene.test_data:
                tasks.append(RenderTask(
                    split='test',
                    camera=test_camera,
                    camera_label=test_camera.name,
                    blender_frame=blender_frame,
                    output_index=output_index,
                    filepath=_render_filepath(_camera_root(output_path, 'test'), output_index),
                    matrix_world=test_matrices[output_index].copy(),
                ))

        if scene.logs:
            _write_log(scene, output_path, len(frames))
    except Exception:
        scene.save_path = requested_save_path
        if initial_state is not None:
            restore_scene(scene, initial_state)
        raise

    return PreparedExport(
        output_path=output_path,
        frames=frames,
        tasks=tasks,
        initial_state=initial_state,
        train_cameras=train_cameras,
        test_camera=test_camera,
        calibration=calibration,
    )


def restore_scene(scene, initial_state):
    errors = []

    def restore_attribute(target, attribute, value):
        try:
            setattr(target, attribute, value)
        except Exception as exception:
            errors.append('{}: {}'.format(attribute, exception))

    restore_attribute(scene.render, 'filepath', initial_state['render_filepath'])
    restore_attribute(scene.render.image_settings, 'file_format', initial_state['file_format'])
    restore_attribute(scene.render.image_settings, 'color_mode', initial_state['color_mode'])
    restore_attribute(scene.render, 'use_file_extension', initial_state['use_file_extension'])
    restore_attribute(scene.render, 'resolution_x', initial_state['resolution_x'])
    restore_attribute(scene.render, 'resolution_y', initial_state['resolution_y'])
    restore_attribute(scene.render, 'resolution_percentage', initial_state['resolution_percentage'])
    restore_attribute(scene.render, 'pixel_aspect_x', initial_state['pixel_aspect_x'])
    restore_attribute(scene.render, 'pixel_aspect_y', initial_state['pixel_aspect_y'])
    camera_data = initial_state.get('camera_data')
    if camera_data is None and carr_rig.CARR_TEST_NAME in scene.objects:
        camera_data = scene.objects[carr_rig.CARR_TEST_NAME].data
    if camera_data is not None:
        restore_attribute(camera_data, 'lens', initial_state['lens'])
        restore_attribute(camera_data, 'sensor_width', initial_state['sensor_width'])
        restore_attribute(camera_data, 'sensor_height', initial_state['sensor_height'])
        restore_attribute(camera_data, 'sensor_fit', initial_state['sensor_fit'])
    try:
        restore_attribute(scene, 'camera', initial_state['camera'])
    finally:
        try:
            scene.frame_set(initial_state['frame'])
        except Exception as exception:
            errors.append('frame: {}'.format(exception))
    if errors:
        raise RuntimeError('; '.join(errors))


def configure_rgba(scene):
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.use_file_extension = True


def apply_calibration(scene, camera, calibration):
    camera.data.lens = calibration.lens
    camera.data.sensor_width = calibration.sensor_width
    camera.data.sensor_height = calibration.sensor_height
    camera.data.sensor_fit = calibration.sensor_fit
    scene.render.resolution_x = calibration.resolution_x
    scene.render.resolution_y = calibration.resolution_y
    scene.render.resolution_percentage = calibration.resolution_percentage
    scene.render.pixel_aspect_x = calibration.pixel_aspect_x
    scene.render.pixel_aspect_y = calibration.pixel_aspect_y


def render_task(context, task, calibration):
    scene = context.scene
    scene.frame_set(task.blender_frame)
    apply_calibration(scene, task.camera, calibration)
    task.camera.matrix_world = task.matrix_world.copy()
    context.view_layer.update()
    scene.camera = task.camera
    scene.render.filepath = task.filepath
    return bpy.ops.render.render('EXEC_DEFAULT', write_still=True)


def complete_metadata_only(context, prepared):
    restore_scene(context.scene, prepared.initial_state)
    return {'FINISHED'}


class CameraArray(bpy.types.Operator):
    bl_idname = 'object.camera_array'
    bl_label = 'Camera Array CArr'
    _is_running = False

    @classmethod
    def poll(cls, context):
        return not cls._is_running

    def execute(self, context):
        try:
            prepared = prepare_export(context)
        except CArrValidationError as exception:
            self.report({'ERROR'}, str(exception))
            return {'CANCELLED'}
        except Exception as exception:
            self.report({'ERROR'}, 'CArr export failed: {}'.format(exception))
            return {'CANCELLED'}
        if not context.scene.render_frames:
            try:
                complete_metadata_only(context, prepared)
            except Exception as exception:
                self.report({'ERROR'}, 'CArr scene restoration failed: {}'.format(exception))
                return {'CANCELLED'}
            self.report({'INFO'}, 'CArr dataset saved to {}.'.format(prepared.output_path))
            return {'FINISHED'}
        return self._start_queue(context, prepared)

    def _start_queue(self, context, prepared):
        self._prepared = prepared
        self._render_index = 0
        self._finalized = False
        self._timer = None
        self._progress_started = False

        type(self)._is_running = True
        try:
            configure_rgba(context.scene)
            context.window_manager.progress_begin(0, len(prepared.tasks))
            self._progress_started = True
            self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
            context.window_manager.modal_handler_add(self)
        except Exception as exception:
            self.report({'ERROR'}, 'CArr render setup failed: {}'.format(exception))
            return self._finish(context, False)

        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'ESC':
            return self._finish(context, False)

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        if self._render_index >= len(self._prepared.tasks):
            return self._finish(context, True)

        task = self._prepared.tasks[self._render_index]
        try:
            render_result = render_task(context, task, self._prepared.calibration)
        except Exception as exception:
            self.report(
                {'ERROR'},
                'CArr render failed for camera {} at Blender frame {}: {}'.format(
                    task.camera_label, task.blender_frame, exception
                ),
            )
            return self._finish(context, False)

        if 'CANCELLED' in render_result:
            self.report(
                {'ERROR'},
                'Blender cancelled CArr render for camera {} at Blender frame {}.'.format(
                    task.camera_label, task.blender_frame
                ),
            )
            return self._finish(context, False)

        self._render_index += 1
        context.window_manager.progress_update(self._render_index)

        if context.screen is not None:
            for area in context.screen.areas:
                area.tag_redraw()

        if self._render_index >= len(self._prepared.tasks):
            return self._finish(context, True)

        return {'RUNNING_MODAL'}

    def _finish(self, context, success):
        if self._finalized:
            return {'FINISHED'} if success else {'CANCELLED'}

        self._finalized = True
        type(self)._is_running = False

        if self._timer is not None:
            timer = self._timer
            self._timer = None
            try:
                context.window_manager.event_timer_remove(timer)
            except Exception:
                pass

        if self._progress_started:
            self._progress_started = False
            try:
                context.window_manager.progress_end()
            except Exception:
                pass

        restoration_error = None
        try:
            restore_scene(context.scene, self._prepared.initial_state)
        except Exception as exception:
            restoration_error = exception

        if restoration_error is not None:
            self.report(
                {'ERROR'},
                'CArr scene restoration failed: {}. Partial files remain in {}.'.format(
                    restoration_error, self._prepared.output_path
                ),
            )
            return {'CANCELLED'}

        if success:
            self.report({'INFO'}, 'CArr dataset saved to {}.'.format(self._prepared.output_path))
            return {'FINISHED'}

        self.report(
            {'WARNING'},
            'CArr export stopped. Partial files remain in {}.'.format(self._prepared.output_path),
        )
        return {'CANCELLED'}
