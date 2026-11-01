"""Finalize the checked side solver and verify an actual normal stream restart."""
import bpy,json,importlib,hashlib
from pathlib import Path
from mathutils import Matrix
from bl_ext.user_default.posecap.posecap_addon import matrix_retarget,stream_operators
from bl_ext.user_default.posecap.posecap_addon.binding_state import load_binding
from bl_ext.user_default.posecap.posecap_addon.stream_session import active_stream_session
from bl_ext.user_default.posecap.posecap_addon.stream_properties import resolve_selected_backend

folder=Path(__file__).resolve().parent;scene=bpy.context.scene;s=scene.posecap;arm=s.target_armature
assert len(load_binding(arm).bones)==52
assert 'fahad_rest_compensated_binding' in Path(matrix_retarget.__file__).read_text(encoding='utf-8'), 'Restart patch not visible to actual Blender.'
importlib.reload(matrix_retarget)
stream_operators.requires_matrix_retarget=matrix_retarget.requires_matrix_retarget
if s.lifecycle_state!='STOPPED':bpy.ops.posecap.stop_stream()
arm['fahad_rest_compensated_binding']=True
assert not matrix_retarget.requires_matrix_retarget(arm,load_binding(arm))
s.pose_backend_id='fahad_live';assert resolve_selected_backend(s).id=='fahad_live'
assert '--profile-test' not in resolve_selected_backend(s).command, 'Temporary test manifest was not restored.'
reports={name:json.loads((folder/(name+'.json')).read_text(encoding='utf-8')) for name in
         ['side_verification','side_mesh_verification','side_live_verification','side_restart_verification']}
scene['QA_scope']='Side-profile update: 787 actual applied camera frames exactly matched replay; observed +/-90-degree side poses rendered. Ten synthetic side cases, six independent-leg cases, 511 previous frames and 163 restart poses checked. Multi-person/framing issues prevent an all-poses live guarantee.'
scene['Fahad_solver_version']='side-profile-1'
scene['Fahad_side_validation']=scene['QA_scope']
text=bpy.data.texts.get('Fahad Side - Verification') or bpy.data.texts.new('Fahad Side - Verification')
text.clear();text.write(json.dumps(reports,indent=2))

# Start from the last captured bent pose, deliberately NOT a reset T pose.
result=bpy.ops.posecap.start_stream()
assert 'FINISHED' in result,s.status_message
session=active_stream_session();writer=type(session._timer._writer).__name__
assert writer=='LiveTargetArmaturePoseWriter',writer

def finish():
    bpy.ops.posecap.stop_stream()
    for b in arm.pose.bones:b.matrix_basis=Matrix.Identity(4)
    bpy.context.view_layer.update()
    s.status_message='Side-profile fix checked. Start Stream; one person, full body, calibrate facing front.'
    report={'actual_normal_restart_writer':writer,'temporary_test_flags_removed':True,
            'backend':resolve_selected_backend(s).id,'solver_sha256':hashlib.sha256((folder/'fahad_backend.py').read_bytes()).hexdigest()}
    (folder/'side_final_session.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    scene['Fahad_restart_checked']=True
    bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
    print('SIDE_FIX_SAVED',bpy.data.filepath,report)
    return None

bpy.app.timers.register(finish,first_interval=5.)
print('Normal Stop/Start check passed with direct binding; saving neutral startup in 5 seconds.')
