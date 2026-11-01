"""Open safe Studio controls; never start a second camera automatically."""
import sys
from pathlib import Path
import bpy
sys.path.insert(0,str(Path(__file__).resolve().parent))
# A user may save while the separate garment preview scene is open.
# Register AR against its own scene, never against the preview scene.
studio_scenes=[s for s in bpy.data.scenes if s.get('GS_studio_version')]
if len(studio_scenes)==1 and bpy.context.window:
    bpy.context.window.scene=studio_scenes[0]
import ar_runtime
ar_runtime.register()
import garment_studio
garment_studio.register(ar_runtime)
bpy.data.objects['Armature'].data.pose_position='POSE'
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type=='VIEW_3D':
            area.spaces.active.show_region_ui=True
            area.spaces.active.region_3d.view_perspective='CAMERA'
            area.spaces.active.shading.type='MATERIAL'
            area.spaces.active.overlay.show_overlays=False
print('GARMENT_STUDIO_READY: N > Garments. Original AR folder untouched.')
