"""Restart the actual session with the wrist-roll update; preserve rig binding."""
import bpy
from pathlib import Path
from mathutils import Matrix
from bl_ext.user_default.posecap.posecap_addon.binding_state import load_binding
from bl_ext.user_default.posecap.posecap_addon.stream_properties import resolve_selected_backend

scene=bpy.context.scene
assert resolve_selected_backend(scene.posecap).id=='fahad_live'
if scene.posecap.lifecycle_state!='STOPPED':bpy.ops.posecap.stop_stream()
arm=scene.posecap.target_armature
assert len(load_binding(arm).bones)==52
for bone in arm.pose.bones:bone.matrix_basis=Matrix.Identity(4)
bpy.context.view_layer.update()
scene['Fahad_user_validation']='User confirmed head and legs are correct in the interactive session. Added palm-driven forearm roll and wrist bend only.'
scene.posecap.status_message='Wrist/forearm update ready; head and legs solver unchanged.'
# Save a neutral startup, not the last transient tracking pose.
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)

def resume():
    print('Fahad Live forearm/wrist update: starting existing backend with revised solver.')
    bpy.ops.posecap.start_stream()
    return None
bpy.app.timers.register(resume,first_interval=6.)
print('Restart in 6 seconds. Look straight for head calibration, then rotate palms up/down.')
