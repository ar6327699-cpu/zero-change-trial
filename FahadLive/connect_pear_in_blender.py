"""Select the installed PoseCap PEAR backend for the 'Armature' in this Blender session.

Run from Blender's Text Editor (Run Script) while capture is STOPPED.
This does not start the camera, does not save the .blend, and does not touch
the stored Mixamo binding or the fahad_live (MediaPipe) backend. To go back,
set Pose Backend to "Fahad Live - Calibrated Mixamo + Fingers" in the panel
or run connect_in_blender.py.
"""
import bpy

from bl_ext.user_default.posecap.posecap_addon import backend_registry
from bl_ext.user_default.posecap.posecap_addon.binding_state import load_binding
from bl_ext.user_default.posecap.posecap_addon.stream_properties import resolve_selected_backend
import os

scene = bpy.context.scene
settings = scene.posecap
arm = bpy.data.objects.get('Armature')
assert arm is not None and arm.type == 'ARMATURE', "Armature object not found."
assert settings.lifecycle_state == 'STOPPED', 'Stop capture before switching backend.'

catalog = backend_registry.discover_installed_pose_backends(dict(os.environ))
ready_ids = [b.manifest.id for b in catalog.ready]
assert 'pear' in ready_ids, f"PEAR backend not ready. Ready: {ready_ids}; issues: {catalog.issues}"

binding = load_binding(arm)
assert binding is not None and len(binding.bones) > 0, "Run Convert Character for PoseCap first."

settings.target_armature = arm
settings.pose_backend_id = 'pear'
settings.apply_orientation_fix = True      # PEAR-to-SMPL-X orientation correction
settings.pose_smoothing = True             # PEAR is noisier than the MediaPipe solver
settings.camera_pitch = 0
settings.apply_arms = settings.apply_legs = settings.apply_torso = True
settings.world_position_experimental = False
settings.record_live_mocap = False
settings.preview_enabled = True
settings.status_message = 'PEAR selected. Ready for live test.'

assert resolve_selected_backend(settings).id == 'pear'
print('PEAR_SELECTED', 'bones=' + str(len(binding.bones)), 'ready=' + ','.join(ready_ids))
