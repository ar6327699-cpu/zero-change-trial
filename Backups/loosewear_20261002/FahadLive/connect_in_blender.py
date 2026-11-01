"""Register the existing local backend from the actual interactive Blender process.

Run in Blender, not a separate background process. Preserve the live unsaved
scene first. Do not start the camera or overwrite the working blend here.
"""
import bpy
import json
import os
from pathlib import Path
from datetime import datetime
from mathutils import Matrix, Quaternion

from bl_ext.user_default.posecap.posecap_addon import backend_registry
from bl_ext.user_default.posecap.posecap_addon.character_setup import build_pose_binding, mixamo_preset
from bl_ext.user_default.posecap.posecap_addon.binding_state import store_binding, load_binding
from bl_ext.user_default.posecap.posecap_addon.stream_properties import resolve_selected_backend

folder = Path(__file__).resolve().parent
scene = bpy.context.scene
arm = bpy.data.objects.get('Armature')
assert arm is not None and arm.type == 'ARMATURE'
assert scene.posecap.lifecycle_state == 'STOPPED', 'Stop capture before reconnecting.'
assert arm.animation_data is None, 'Unexpected user animation; do not discard it.'
paths = backend_registry.default_installation_paths(dict(os.environ))
assert paths is not None
registry = paths.backend_registry
python = registry / 'mediapipe/runtime/Scripts/python.exe'
model = registry / 'mediapipe/models/holistic_landmarker.task'
script = folder / 'fahad_backend.py'
assert python.is_file() and model.is_file() and script.is_file(), 'Required runtime/model/script missing in interactive Blender.'
compile(script.read_text(encoding='utf-8'), str(script), 'exec')
manifest = {
    'schema_version': 1, 'id': 'fahad_live',
    'display_name': 'Fahad Live - Calibrated Mixamo + Fingers',
    'command': [str(python), str(script), 'live', '--model-path', str(model)],
    'protocol_versions': [1], 'capabilities': ['body', 'hands'],
    'requires_body_models': False, 'apply_orientation_fix': False,
    'compatibility': {'operating_systems': ['windows'], 'accelerators': ['cpu'],
                      'account': 'No account required', 'license': 'Local per-character MediaPipe solver'}
}
# The registry visible to the interactive app is the authoritative target.
backend_folder = registry / 'fahad_live'
backend_folder.mkdir(parents=True, exist_ok=True)
manifest_path = backend_folder / 'backend.json'
encoded = json.dumps(manifest, indent=2)
if manifest_path.exists() and manifest_path.read_text(encoding='utf-8') != encoded:
    recovery = folder / ('backend_before_connect_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json')
    recovery.write_text(manifest_path.read_text(encoding='utf-8'), encoding='utf-8')
manifest_path.write_text(encoded, encoding='utf-8')
catalog = backend_registry.discover_installed_pose_backends(dict(os.environ))
assert any(b.manifest.id == 'fahad_live' for b in catalog.ready), catalog.issues

backup = folder.parent / ('fahadGenius_session_before_connect_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.blend')
bpy.ops.wm.save_as_mainfile(filepath=str(backup), copy=True)
for bone in arm.pose.bones:
    bone.matrix_basis = Matrix.Identity(4)
    bone.rotation_mode = 'QUATERNION'
bpy.context.view_layer.update()
binding = build_pose_binding(arm, mixamo_preset('mixamorig:', include_hands=True))
store_binding(arm, binding)
assert len(load_binding(arm).bones) == 52
settings = scene.posecap
settings.target_armature = arm
settings.pose_backend_id = 'fahad_live'
settings.apply_orientation_fix = False
settings.pose_smoothing = False
settings.camera_pitch = 0
settings.apply_arms = settings.apply_legs = settings.apply_torso = True
settings.world_position_experimental = False
settings.record_live_mocap = False
settings.preview_enabled = True
settings.capture_width = 1280
settings.capture_height = 720
settings.status_message = 'Fahad Live connected in this Blender session. Ready for actual viewport test.'
assert resolve_selected_backend(settings).id == 'fahad_live'
for screen in bpy.data.screens:
    for area in screen.areas:
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.region_3d.view_rotation = Quaternion((1, 0, 0), 1.5707963267948966)
                space.region_3d.view_location = (0, 0, 1.9)
                space.region_3d.view_distance = 5.7
                space.overlay.show_overlays = False
        area.tag_redraw()
print('INTERACTIVE_CONNECTED', 'backend=fahad_live', 'joints=52', 'backup=' + str(backup), 'working=' + bpy.data.filepath)
