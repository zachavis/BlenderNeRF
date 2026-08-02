import math
import os
import shutil
import bpy
from . import helper, blender_nerf_operator


EMPTY_NAME = 'BlenderNeRF Sphere'
CAMERA_NAME = 'BlenderNeRF Camera'
FIXED_CAMERA_NAME = 'BlenderNeRF Fixed Camera'


class CameraOnSphere(blender_nerf_operator.BlenderNeRF_Operator):
    '''Camera on Sphere Operator'''
    bl_idname = 'object.camera_on_sphere'
    bl_label = 'Camera on Sphere COS'
    _is_running = False

    @classmethod
    def poll(cls, context):
        return not cls._is_running

    def uniformly_spaced_frames(self, frame_start, frame_end, count):
        total = frame_end - frame_start + 1
        count = min(max(count, 1), total)

        if count == 1:
            return [frame_start]

        span = total - 1
        return [
            frame_start + int(math.floor(index * span / (count - 1) + 0.5))
            for index in range(count)
        ]

    def normalized_time(self, frame, frame_start, frame_end):
        if frame_end == frame_start:
            return 0.0
        return (frame - frame_start) / (frame_end - frame_start)

    def evaluate_sphere_camera(self, context, scene, camera, frame, seed):
        scene.seed = seed
        scene.frame_set(frame)
        camera.location = helper.sample_from_sphere(scene)
        context.view_layer.update()

    def create_fixed_camera(self, scene, source_camera, transform_matrix):
        camera_data = source_camera.data.copy()
        camera = bpy.data.objects.new(FIXED_CAMERA_NAME, camera_data)
        scene.collection.objects.link(camera)
        camera.matrix_world = transform_matrix
        return camera

    def delete_temporary_camera(self, camera):
        camera_data = camera.data
        bpy.data.objects.remove(camera, do_unlink=True)
        if camera_data.users == 0:
            bpy.data.cameras.remove(camera_data)

    def output_frame_path(self, scene, split, output_index):
        stem = 'r_{:03d}'.format(output_index)
        extension = scene.render.file_extension if scene.render.use_file_extension else ''
        json_name = stem if (scene.nerf or scene.splats) else stem + extension
        json_path = self.format_dataset_path(scene, split, json_name, relative_prefix=True)
        return stem, json_path

    def prepare_split(self, context, scene, output_path, split, frames, camera, seed=None, fixed_matrix=None):
        split_path = os.path.join(output_path, split)
        os.makedirs(split_path, exist_ok=True)
        output_data = self.get_camera_intrinsics(scene, camera)
        frame_data = []
        render_tasks = []

        if not (scene.splats and scene.splats_test_dummy and split == 'test'):
            for output_index, frame in enumerate(frames):
                if seed is None:
                    scene.frame_set(frame)
                    camera.matrix_world = fixed_matrix
                    context.view_layer.update()
                else:
                    self.evaluate_sphere_camera(context, scene, camera, frame, seed)

                scene.camera = camera
                stem, json_path = self.output_frame_path(scene, split, output_index)
                frame_data.append({
                    'file_path': json_path,
                    'time': self.normalized_time(frame, scene.frame_start, scene.frame_end),
                    'transform_matrix': self.listify_matrix(camera.matrix_world)
                })

                if scene.render_frames:
                    render_tasks.append({
                        'camera': camera,
                        'filepath': os.path.join(split_path, stem),
                        'fixed_matrix': fixed_matrix.copy() if fixed_matrix is not None else None,
                        'frame': frame,
                        'seed': seed
                    })

        output_data['frames'] = frame_data
        self.save_json(output_path, 'transforms_{}.json'.format(split), output_data)
        return render_tasks

    def ensure_sphere_camera(self, context, scene):
        if EMPTY_NAME not in scene.objects:
            if scene.sphere_exists:
                scene.sphere_exists = False
            if not scene.show_sphere:
                scene.show_sphere = True
            else:
                helper.visualize_sphere(scene, context)

        if CAMERA_NAME not in scene.objects:
            if scene.camera_exists:
                scene.camera_exists = False
            if not scene.show_camera:
                scene.show_camera = True
            else:
                helper.visualize_camera(scene, context)
        return scene.objects[CAMERA_NAME]

    def clean_generated_output(self, output_path):
        for split in ('train', 'val', 'test', 'fixed'):
            split_path = os.path.join(output_path, split)
            transforms_path = os.path.join(output_path, 'transforms_{}.json'.format(split))
            if os.path.isdir(split_path):
                shutil.rmtree(split_path)
            if os.path.isfile(transforms_path):
                os.remove(transforms_path)

        for filename in ('log.txt', 'points3d.ply'):
            filepath = os.path.join(output_path, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)

    def restore_scene(self, scene, initial_state, fixed_camera):
        scene.seed = initial_state['seed']
        scene.render.filepath = initial_state['render_filepath']
        scene.camera = initial_state['camera']

        if fixed_camera is not None:
            self.delete_temporary_camera(fixed_camera)

        if not initial_state['camera_exists'] and CAMERA_NAME in scene.objects:
            helper.delete_camera(scene, CAMERA_NAME)

        if not initial_state['sphere_exists'] and EMPTY_NAME in scene.objects:
            bpy.data.objects.remove(scene.objects[EMPTY_NAME], do_unlink=True)
            scene.show_sphere = False
            scene.sphere_exists = False

        scene.frame_set(initial_state['frame'])

    def archive_output(self, output_path):
        shutil.make_archive(output_path, 'zip', output_path)
        shutil.rmtree(output_path)

    def render_frame(self, context):
        task = self._render_tasks[self._render_index]

        if task['seed'] is None:
            self._scene.frame_set(task['frame'])
            task['camera'].matrix_world = task['fixed_matrix']
            context.view_layer.update()
        else:
            self.evaluate_sphere_camera(
                context,
                self._scene,
                task['camera'],
                task['frame'],
                task['seed']
            )

        self._scene.camera = task['camera']
        self._scene.render.filepath = task['filepath']
        return bpy.ops.render.render('EXEC_DEFAULT', write_still=True)

    def finish_render_queue(self, context, success, message=None):
        if self._finalized:
            return {'FINISHED'} if success else {'CANCELLED'}

        self._finalized = True
        type(self)._is_running = False

        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

        context.window_manager.progress_end()
        self.restore_scene(self._scene, self._initial_state, self._fixed_camera)

        if success:
            try:
                self.archive_output(self._output_path)
            except Exception as exception:
                self.report({'ERROR'}, 'COS archive failed: {}'.format(exception))
                return {'CANCELLED'}

            self.report({'INFO'}, 'COS dataset saved to {}.zip'.format(self._output_path))
            return {'FINISHED'}

        if message is not None:
            self.report({'WARNING'}, message)
        return {'CANCELLED'}

    def modal(self, context, event):
        if event.type == 'ESC':
            return self.finish_render_queue(
                context,
                False,
                'COS export cancelled. Partial files remain in {}.'.format(self._output_path)
            )

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        if self._render_index >= len(self._render_tasks):
            return self.finish_render_queue(context, True)

        try:
            render_result = self.render_frame(context)
        except Exception as exception:
            return self.finish_render_queue(
                context,
                False,
                'COS render failed: {}. Partial files remain in {}.'.format(
                    exception,
                    self._output_path
                )
            )

        if 'CANCELLED' in render_result:
            return self.finish_render_queue(
                context,
                False,
                'Blender cancelled render {} of {}. Partial files remain in {}.'.format(
                    self._render_index + 1,
                    len(self._render_tasks),
                    self._output_path
                )
            )

        self._render_index += 1
        context.window_manager.progress_update(self._render_index)

        if context.screen is not None:
            for area in context.screen.areas:
                area.tag_redraw()

        if self._render_index >= len(self._render_tasks):
            return self.finish_render_queue(context, True)

        return {'RUNNING_MODAL'}

    def start_render_queue(self, context, scene, output_path, initial_state, fixed_camera, render_tasks):
        self._scene = scene
        self._output_path = output_path
        self._initial_state = initial_state
        self._fixed_camera = fixed_camera
        self._render_tasks = render_tasks
        self._render_index = 0
        self._finalized = False
        self._timer = None

        type(self)._is_running = True
        context.window_manager.progress_begin(0, len(render_tasks))
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)

        return {'RUNNING_MODAL'}

    def execute(self, context):
        scene = context.scene

        error_messages = self.asserts(scene, method='COS')
        if error_messages:
            self.report({'ERROR'}, error_messages[0])
            return {'CANCELLED'}

        output_dir = bpy.path.clean_name(scene.cos_dataset_name)
        output_path = os.path.join(scene.save_path, output_dir)
        os.makedirs(output_path, exist_ok=True)
        self.clean_generated_output(output_path)

        initial_state = {
            'camera': scene.camera,
            'camera_exists': CAMERA_NAME in scene.objects,
            'sphere_exists': EMPTY_NAME in scene.objects,
            'frame': scene.frame_current,
            'render_filepath': scene.render.filepath,
            'seed': scene.seed
        }

        base_seed = scene.seed
        train_frames = list(range(scene.frame_start, scene.frame_end + 1))
        eval_frames = self.uniformly_spaced_frames(
            scene.frame_start,
            scene.frame_end,
            scene.cos_eval_frames
        )
        fixed_camera = None
        export_error = None
        render_tasks = []

        try:
            if scene.logs:
                self.save_log_file(scene, output_path, method='COS')
            if scene.splats:
                self.save_splats_ply(scene, output_path)

            needs_sphere_camera = (
                scene.train_data
                or scene.cos_val_data
                or scene.test_data
                or (scene.cos_fixed_data and scene.cos_fixed_camera_mode == 'TRAIN_VIEW')
            )
            sphere_camera = self.ensure_sphere_camera(context, scene) if needs_sphere_camera else None

            if scene.cos_fixed_data:
                if scene.cos_fixed_camera_mode == 'TRAIN_VIEW':
                    fixed_view = min(scene.cos_fixed_train_view, len(train_frames) - 1)
                    if fixed_view != scene.cos_fixed_train_view:
                        self.report({'WARNING'}, 'Fixed train view exceeds the animation range and was clamped.')
                    self.evaluate_sphere_camera(
                        context,
                        scene,
                        sphere_camera,
                        train_frames[fixed_view],
                        base_seed
                    )
                    fixed_source = sphere_camera
                else:
                    fixed_source = scene.camera_fixed_target
                    context.view_layer.update()

                fixed_camera = self.create_fixed_camera(
                    scene,
                    fixed_source,
                    fixed_source.matrix_world.copy()
                )

            if scene.train_data:
                render_tasks.extend(self.prepare_split(
                    context, scene, output_path, 'train', train_frames,
                    sphere_camera, seed=base_seed
                ))

            if scene.cos_val_data:
                render_tasks.extend(self.prepare_split(
                    context, scene, output_path, 'val', eval_frames,
                    sphere_camera, seed=scene.cos_val_seed
                ))

            if scene.test_data:
                render_tasks.extend(self.prepare_split(
                    context, scene, output_path, 'test', eval_frames,
                    sphere_camera, seed=scene.cos_test_seed
                ))

            if scene.cos_fixed_data:
                fixed_matrix = fixed_camera.matrix_world.copy()
                render_tasks.extend(self.prepare_split(
                    context, scene, output_path, 'fixed', train_frames,
                    fixed_camera, fixed_matrix=fixed_matrix
                ))
        except Exception as exception:
            export_error = exception

        if export_error is not None:
            self.restore_scene(scene, initial_state, fixed_camera)
            self.report({'ERROR'}, 'COS export failed: {}'.format(export_error))
            return {'CANCELLED'}

        if not render_tasks:
            self.restore_scene(scene, initial_state, fixed_camera)
            try:
                self.archive_output(output_path)
            except Exception as exception:
                self.report({'ERROR'}, 'COS archive failed: {}'.format(exception))
                return {'CANCELLED'}

            self.report({'INFO'}, 'COS dataset saved to {}.zip'.format(output_path))
            return {'FINISHED'}

        return self.start_render_queue(
            context,
            scene,
            output_path,
            initial_state,
            fixed_camera,
            render_tasks
        )
