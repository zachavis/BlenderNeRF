import bpy
from . import helper, blender_nerf_ui, sof_ui, ttc_ui, cos_ui, sof_operator, ttc_operator, cos_operator, carr_operator, carr_rig


# blender info
bl_info = {
    'name': 'BlenderNeRF',
    'description': 'Easy NeRF synthetic dataset creation within Blender',
    'author': 'Maxime Raafat',
    'version': (6, 1, 0),
    'blender': (4, 2, 0),
    'location': '3D View > N panel > BlenderNeRF',
    'doc_url': 'https://github.com/maximeraafat/BlenderNeRF',
    'category': 'Object',
}

# global addon script variables
TRAIN_CAM = 'Train Cam'
TEST_CAM = 'Test Cam'
VERSION = '.'.join(str(x) for x in bl_info['version'])

# addon blender properties
PROPS = [
    # global controllable properties
    ('train_data', bpy.props.BoolProperty(name='Train', description='Construct the training data', default=True) ),
    ('test_data', bpy.props.BoolProperty(name='Test', description='Construct the testing data', default=True) ),
    ('aabb', bpy.props.IntProperty(name='AABB', description='AABB scale as defined in Instant NGP', default=4, soft_min=1, soft_max=128) ),
    ('render_frames', bpy.props.BoolProperty(name='Render Frames', description='Whether enabled split frames should be rendered. If not selected, only metadata files will be generated', default=True) ),
    ('logs', bpy.props.BoolProperty(name='Save Log File', description='Whether to create a log file containing information on the BlenderNeRF run', default=False) ),
    ('splats', bpy.props.BoolProperty(name='Gaussian Points', description='Whether to export a points3d.ply file for Gaussian Splatting', default=False) ),
    ('splats_test_dummy', bpy.props.BoolProperty(name='Dummy Test Camera', description='Whether to export a dummy test transforms.json file or the full set of test camera poses', default=True) ),
    ('nerf', bpy.props.BoolProperty(name='NeRF', description='Whether to export the camera transforms.json files in the defaut NeRF file format convention', default=False) ),
    ('path_format', bpy.props.EnumProperty(
        name='Path Format',
        description='Path separator convention used in transforms JSON files',
        items=(
            ('POSIX', 'Linux', 'Use forward slashes in dataset paths'),
            ('WINDOWS', 'Windows', 'Use backslashes in dataset paths'),
        ),
        default='POSIX'
    ) ),
    ('save_path', bpy.props.StringProperty(name='Save Path', description='Path to the output directory in which the synthetic dataset will be stored', subtype='DIR_PATH') ),

    # global automatic properties
    ('init_frame_step', bpy.props.IntProperty(name='Initial Frame Step') ),
    ('init_output_path', bpy.props.StringProperty(name='Initial Output Path', subtype='DIR_PATH') ),
    ('rendering', bpy.props.BoolVectorProperty(name='Rendering', description='Whether one of the SOF, TTC or COS methods is rendering', default=(False, False, False), size=3) ),
    ('blendernerf_version', bpy.props.StringProperty(name='BlenderNeRF Version', default=VERSION) ),

    # sof properties
    ('sof_dataset_name', bpy.props.StringProperty(name='Name', description='Name of the SOF dataset : the data will be stored under <save path>/<name>', default='dataset') ),
    ('train_frame_steps', bpy.props.IntProperty(name='Frame Step', description='Frame step N for the captured training frames. Every N-th frame will be used for training NeRF', default=3, soft_min=1) ),

    # ttc properties
    ('ttc_dataset_name', bpy.props.StringProperty(name='Name', description='Name of the TTC dataset : the data will be stored under <save path>/<name>', default='dataset') ),
    ('ttc_nb_frames', bpy.props.IntProperty(name='Frames', description='Number of training frames from the training camera', default=100, soft_min=1) ),
    ('camera_train_target', bpy.props.PointerProperty(type=bpy.types.Object, name=TRAIN_CAM, description='Pointer to the training camera', poll=helper.poll_is_camera) ),
    ('camera_test_target', bpy.props.PointerProperty(type=bpy.types.Object, name=TEST_CAM, description='Pointer to the testing camera', poll=helper.poll_is_camera) ),

    # cos controllable properties
    ('cos_dataset_name', bpy.props.StringProperty(name='Name', description='Name of the COS dataset : the data will be stored under <save path>/<name>', default='dataset') ),
    ('sphere_location', bpy.props.FloatVectorProperty(name='Location', description='Center position of the training sphere', unit='LENGTH', update=helper.properties_ui_upd) ),
    ('sphere_rotation', bpy.props.FloatVectorProperty(name='Rotation', description='Rotation of the training sphere', unit='ROTATION', update=helper.properties_ui_upd) ),
    ('sphere_scale', bpy.props.FloatVectorProperty(name='Scale', description='Scale of the training sphere in xyz axes', default=(1.0, 1.0, 1.0), update=helper.properties_ui_upd) ),
    ('sphere_radius', bpy.props.FloatProperty(name='Radius', description='Radius scale of the training sphere', default=4.0, soft_min=0.01, unit='LENGTH', update=helper.properties_ui_upd) ),
    ('focal', bpy.props.FloatProperty(name='Lens', description='Focal length of the training camera', default=50, soft_min=1, soft_max=5000, unit='CAMERA', update=helper.properties_ui_upd) ),
    ('seed', bpy.props.IntProperty(name='Train Seed', description='Random seed for sampling COS training views', default=0) ),
    ('cos_val_seed', bpy.props.IntProperty(name='Val Seed', description='Random seed for sampling COS validation views', default=100) ),
    ('cos_test_seed', bpy.props.IntProperty(name='Test Seed', description='Random seed for sampling COS test views', default=200) ),
    ('cos_val_data', bpy.props.BoolProperty(name='Val', description='Construct and render the COS validation split', default=True) ),
    ('cos_fixed_data', bpy.props.BoolProperty(name='Fixed', description='Construct and render the COS fixed-camera diagnostic split', default=True) ),
    ('cos_eval_frames', bpy.props.IntProperty(name='Val/Test Frames', description='Number of uniformly selected animation frames used by the COS validation and test splits', default=10, min=1) ),
    ('cos_fixed_camera_mode', bpy.props.EnumProperty(
        name='Fixed Camera',
        description='Source pose for the COS fixed-camera diagnostic split',
        items=(
            ('TRAIN_VIEW', 'Train View', 'Hold one generated COS training view fixed'),
            ('CAMERA', 'Camera', 'Snapshot a separately placed camera'),
        ),
        default='TRAIN_VIEW'
    ) ),
    ('cos_fixed_train_view', bpy.props.IntProperty(name='Train View', description='Zero-based training view index to hold fixed', default=0, min=0) ),
    ('camera_fixed_target', bpy.props.PointerProperty(type=bpy.types.Object, name='Fixed Camera', description='Camera whose current pose and intrinsics will be held fixed', poll=helper.poll_is_camera) ),
    ('show_sphere', bpy.props.BoolProperty(name='Sphere', description='Whether to show the training sphere from which random views will be sampled', default=False, update=helper.visualize_sphere) ),
    ('show_camera', bpy.props.BoolProperty(name='Camera', description='Whether to show the training camera', default=False, update=helper.visualize_camera) ),
    ('upper_views', bpy.props.BoolProperty(name='Upper Views', description='Whether to sample views from the upper hemisphere of the training sphere only', default=False) ),
    ('outwards', bpy.props.BoolProperty(name='Outwards', description='Whether to point the camera outwards of the training sphere', default=False, update=helper.properties_ui_upd) ),

    # carr properties
    ('carr_dataset_name', bpy.props.StringProperty(name='Name', description='Name of the CArr dataset directory', default='dataset')),
    ('carr_geometry', bpy.props.EnumProperty(
        name='Geometry',
        items=(
            ('CIRCLE', 'Circle', 'Equally spaced cameras on the local XY circle'),
            ('HEMISPHERE', 'Hemisphere', 'Near-uniform cameras on the local +Z hemisphere'),
            ('SPHERE', 'Sphere', 'Near-uniform cameras on the full sphere'),
        ),
        default='CIRCLE', update=carr_rig.carr_rig_property_update,
    )),
    ('carr_camera_count', bpy.props.IntProperty(
        name='Camera Count', default=10, min=2,
        description='Number of fixed synchronized training cameras',
        update=carr_rig.carr_rig_property_update,
    )),
    ('carr_location', bpy.props.FloatVectorProperty(
        name='Location', unit='LENGTH', update=carr_rig.carr_rig_property_update,
    )),
    ('carr_rotation', bpy.props.FloatVectorProperty(
        name='Rotation', unit='ROTATION', subtype='EULER', update=carr_rig.carr_rig_property_update,
    )),
    ('carr_radius', bpy.props.FloatProperty(
        name='Radius', default=4.0, min=0.001, unit='LENGTH', update=carr_rig.carr_rig_property_update,
    )),
    ('carr_focal', bpy.props.FloatProperty(
        name='Lens', default=50.0, min=1.0, max=5000.0, unit='CAMERA', update=carr_rig.carr_rig_property_update,
    )),
    ('carr_show_rig', bpy.props.BoolProperty(
        name='Show Rig', default=False, update=carr_rig.carr_show_rig_update,
    )),

    # cos automatic properties
    ('sphere_exists', bpy.props.BoolProperty(name='Sphere Exists', description='Whether the sphere exists', default=False) ),
    ('init_sphere_exists', bpy.props.BoolProperty(name='Init sphere exists', description='Whether the sphere initially exists', default=False) ),
    ('camera_exists', bpy.props.BoolProperty(name='Camera Exists', description='Whether the camera exists', default=False) ),
    ('init_camera_exists', bpy.props.BoolProperty(name='Init camera exists', description='Whether the camera initially exists', default=False) ),
    ('init_active_camera', bpy.props.PointerProperty(type=bpy.types.Object, name='Init active camera', description='Pointer to initial active camera', poll=helper.poll_is_camera) ),
    ('init_frame_end', bpy.props.IntProperty(name='Initial Frame End') ),
]

# classes to register / unregister
CLASSES = [
    blender_nerf_ui.BlenderNeRF_UI,
    sof_ui.SOF_UI,
    ttc_ui.TTC_UI,
    cos_ui.COS_UI,
    sof_operator.SubsetOfFrames,
    ttc_operator.TrainTestCameras,
    cos_operator.CameraOnSphere,
    carr_operator.CameraArray
]

# load addon
def register():
    for (prop_name, prop_value) in PROPS:
        setattr(bpy.types.Scene, prop_name, prop_value)

    for cls in CLASSES:
        bpy.utils.register_class(cls)

    bpy.app.handlers.render_complete.append(helper.post_render)
    bpy.app.handlers.render_cancel.append(helper.post_render)
    bpy.app.handlers.frame_change_post.append(helper.cos_camera_update)
    bpy.app.handlers.depsgraph_update_post.append(helper.properties_desgraph_upd)
    bpy.app.handlers.depsgraph_update_post.append(helper.set_init_props)
    if carr_rig.carr_frame_change not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(carr_rig.carr_frame_change)

# deregister addon
def unregister():
    carr_rig.remove_preview(bpy.context.scene)

    for (prop_name, _) in PROPS:
        delattr(bpy.types.Scene, prop_name)

    bpy.app.handlers.render_complete.remove(helper.post_render)
    bpy.app.handlers.render_cancel.remove(helper.post_render)
    bpy.app.handlers.frame_change_post.remove(helper.cos_camera_update)
    bpy.app.handlers.depsgraph_update_post.remove(helper.properties_desgraph_upd)
    # bpy.app.handlers.depsgraph_update_post.remove(helper.set_init_props)
    if carr_rig.carr_frame_change in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(carr_rig.carr_frame_change)

    for cls in CLASSES:
        bpy.utils.unregister_class(cls)


if __name__ == '__main__':
    register()
