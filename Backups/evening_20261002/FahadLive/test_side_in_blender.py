"""Run one guided 74-second profile test in the actual interactive session.

Keep numerical landmarks and actual applied bone transforms locally. Restore
the normal manifest immediately after launch; do not bake test animation.
"""
import bpy,json,os,time,hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from mathutils import Matrix
from bl_ext.user_default.posecap.posecap_addon import backend_registry
from bl_ext.user_default.posecap.posecap_addon.binding_state import load_binding
from bl_ext.user_default.posecap.posecap_addon.stream_properties import resolve_selected_backend
from bl_ext.user_default.posecap.posecap_addon.stream_session import active_stream_session

folder=Path(__file__).resolve().parent;root=folder.parent.parent;scene=bpy.context.scene
settings=scene.posecap;arm=settings.target_armature
settings.pose_backend_id='fahad_live'
assert resolve_selected_backend(settings).id=='fahad_live'
assert len(load_binding(arm).bones)==52 and arm.animation_data is None
script=folder/'fahad_backend.py';code=script.read_text(encoding='utf-8')
assert "SOLVER_VERSION = 'side-profile-1'" in code
compile(code,str(script),'exec')
if settings.lifecycle_state!='STOPPED':bpy.ops.posecap.stop_stream()
backup=folder.parent/('fahadGenius_before_side_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.blend')
bpy.ops.wm.save_as_mainfile(filepath=str(backup),copy=True)
for bone in arm.pose.bones:bone.matrix_basis=Matrix.Identity(4)
bpy.context.view_layer.update()
settings.apply_orientation_fix=False;settings.pose_smoothing=False
settings.apply_arms=settings.apply_legs=settings.apply_torso=True
settings.world_position_experimental=False;settings.preview_enabled=True
scene['Fahad_solver_version']='side-profile-1'
scene['Fahad_side_validation']='Offline checks passed; guided live side-pose test pending.'
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
registry=backend_registry.default_installation_paths(dict(os.environ)).backend_registry
manifest_path=registry/'fahad_live/backend.json'
normal=manifest_path.read_text(encoding='utf-8');manifest=json.loads(normal)
assert str(script).replace('\\','/').lower() in [str(x).replace('\\','/').lower() for x in manifest['command']]
rows=[];run={'last':-1,'start':0.,'ended':None}

def observe():
    session=active_stream_session()
    # Do not poll client.latest(): it drains the latest-wins mailbox. Observe
    # only AFTER the real timer has applied its frame (installed below).
    finished = session is not None and session._stopped
    if finished or time.monotonic()-run['start']>180:
        (root/'work/side_actual_applied.json').write_text(json.dumps(rows),encoding='utf-8')
        scene['Fahad_side_validation']=f'Guided profile test captured {len(rows)} actual applied frames. Awaiting review.'
        if settings.lifecycle_state!='STOPPED':bpy.ops.posecap.stop_stream()
        print('SIDE_TEST_CAPTURED',len(rows),'actual applied frames; review required.')
        return None
    return .04

def begin():
    test=dict(manifest)
    test['command']=manifest['command']+['--profile-test','--duration','160','--record-landmarks',str(root/'work/side_live_validation.json')]
    manifest_path.write_text(json.dumps(test,indent=2),encoding='utf-8')
    try:
        result=bpy.ops.posecap.start_stream()
        assert 'FINISHED' in result,settings.status_message
    finally:
        manifest_path.write_text(normal,encoding='utf-8')
    run['start']=time.monotonic()
    session=active_stream_session();original_apply=session._timer._apply_frame
    def applied(frame):
        result=original_apply(frame)
        if result and frame.seq!=run['last']:
            run['last']=frame.seq
            rows.append({'frame':asdict(frame),'applied_at':time.time(),
                'bones':{b.name:list(b.rotation_quaternion) for b in arm.pose.bones},
                'writer_type':type(session._timer._writer).__name__})
        return result
    session._timer._apply_frame=applied
    bpy.app.timers.register(observe,first_interval=.05)
    print('SIDE_TEST_STARTED','solver_sha256='+hashlib.sha256(script.read_bytes()).hexdigest())
    return None

bpy.app.timers.register(begin,first_interval=15.)
print('SIDE_TEST_READY: camera starts in 15 seconds. Follow yellow preview instructions. Backup:',backup)
