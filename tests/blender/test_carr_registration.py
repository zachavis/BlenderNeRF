import tomllib
from pathlib import Path
import sys

import bpy

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from carr_test_support import REPO_ROOT, registered_addon


with registered_addon() as addon:
    scene = bpy.context.scene
    assert hasattr(bpy.types, 'VIEW3D_PT_carr_ui')
    assert hasattr(bpy.types, 'OBJECT_OT_camera_array')
    assert scene.carr_geometry == 'CIRCLE'
    assert scene.carr_camera_count == 10
    assert scene.carr_radius == 4.0
    assert scene.carr_focal == 50.0
    assert addon.bl_info['version'] == (6, 2, 0)
    manifest = tomllib.loads((REPO_ROOT / 'blender_manifest.toml').read_text(encoding='utf-8'))
    assert manifest['version'] == '6.2.0'
    assert manifest['blender_version_min'] == '4.2.0'
    assert addon.carr_rig.carr_frame_change in bpy.app.handlers.frame_change_post

print('CArr final registration checks passed')
