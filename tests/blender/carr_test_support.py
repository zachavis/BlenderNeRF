from contextlib import contextmanager
from pathlib import Path
import sys

import bpy


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))


@contextmanager
def registered_addon():
    import BlenderNeRF
    BlenderNeRF.register()
    try:
        yield BlenderNeRF
    finally:
        scene = bpy.context.scene
        if hasattr(scene, 'carr_show_rig'):
            scene.carr_show_rig = False
        BlenderNeRF.unregister()


def matrix_rows(matrix):
    return tuple(tuple(float(value) for value in row) for row in matrix)
