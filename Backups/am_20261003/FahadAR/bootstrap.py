"""Launch the isolated AR experiment with its controls, without saving video."""
from pathlib import Path
import runpy
import bpy

runpy.run_path(str(Path(__file__).with_name('ar_runtime.py')),run_name='__main__')
camera=bpy.context.scene.camera
camera.data.passepartout_alpha=1.0
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type=='VIEW_3D':
            space=area.spaces.active
            space.region_3d.view_perspective='CAMERA'
            space.shading.type='MATERIAL'
            space.overlay.show_overlays=False

def start():
    bpy.ops.fahad_ar.start()
    return None
bpy.app.timers.register(start,first_interval=1.0)
