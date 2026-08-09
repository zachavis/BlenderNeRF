from datetime import datetime
from pathlib import Path
import sys

import bpy


repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root.parent))
import BlenderNeRF

BlenderNeRF.register()
scene = bpy.context.scene
engine_items = scene.render.bl_rna.properties['engine'].enum_items.keys()
scene.render.engine = (
    'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in engine_items else 'BLENDER_EEVEE'
)
scene.render.resolution_x = 64
scene.render.resolution_y = 64
scene.render.resolution_percentage = 100
scene.frame_start = 1
scene.frame_end = 5
scene.carr_geometry = 'CIRCLE'
scene.carr_camera_count = 3
scene.carr_radius = 4.0
scene.train_data = True
scene.test_data = True
scene.render_frames = True
scene.save_path = bpy.app.tempdir
scene.carr_dataset_name = 'carr_modal_manual_' + datetime.now().strftime('%Y%m%d_%H%M%S')
scene.carr_show_rig = True

bpy.ops.object.light_add(type='AREA', location=(2.0, -2.0, 4.0))
bpy.context.object.data.energy = 1000.0
output = Path(scene.save_path) / bpy.path.clean_name(scene.carr_dataset_name)
print('Open the BlenderNeRF N-panel, click PLAY CArr, then press Escape between images.')
print('Expected partial output:', output)
