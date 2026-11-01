"""Short guided test in the user's actual streaming Blender viewport."""
import bpy, blf, time, math, json
from pathlib import Path
from datetime import datetime

scene = bpy.context.scene
arm = scene.posecap.target_armature
assert scene.posecap.lifecycle_state == 'STREAMING'
phases = [('LOOK STRAIGHT - arms relaxed', 7),
          ('Lift ONLY LEFT leg - lower - repeat', 10),
          ('Lift ONLY RIGHT leg - lower - repeat', 10),
          ('Both hands: OPEN - FIST - OPEN', 12)]
start = time.monotonic()
samples = []
current = {'phase': 0, 'remaining': 7}
output = Path(__file__).parent / ('viewport_test_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json')

def draw():
    k=current['phase']
    if k>=len(phases): return
    blf.position(0, 30, 45, 0); blf.size(0, 23); blf.color(0, 1., .9, .15, 1.)
    blf.draw(0, phases[k][0] + '  ' + str(current['remaining']) + 's')

handle=bpy.types.SpaceView3D.draw_handler_add(draw, (), 'WINDOW', 'POST_PIXEL')

def tick():
    elapsed=time.monotonic()-start
    end=0; phase=len(phases)
    for i,(_,seconds) in enumerate(phases):
        end+=seconds
        if elapsed<end: phase=i; break
    current['phase']=phase; current['remaining']=max(0,int(end-elapsed)+1)
    if phase==len(phases) or scene.posecap.lifecycle_state!='STREAMING':
        bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
        output.write_text(json.dumps({'test':'Interactive Blender viewport; user prompted on-screen', 'samples':samples}), encoding='utf-8')
        print('VIEWPORT_TEST_SAVED', str(output), len(samples))
        for screen in bpy.data.screens:
            for area in screen.areas: area.tag_redraw()
        return None
    values={}
    for label,bone in [('left_leg','LeftUpLeg'),('right_leg','RightUpLeg'),('left_index','LeftHandIndex2'),('right_index','RightHandIndex2')]:
        values[label]=math.degrees(arm.pose.bones['mixamorig:'+bone].rotation_quaternion.angle)
    head=arm.pose.bones['mixamorig:Head']
    delta=(arm.matrix_world@head.matrix)@(arm.matrix_world@head.bone.matrix_local).inverted()
    values['head_pitch']=math.degrees(delta.to_euler('XYZ').x)
    samples.append({'t':elapsed,'phase':phase,**values})
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':area.tag_redraw()
    return .1

bpy.app.timers.register(tick, first_interval=.1)
print('VIEWPORT_TEST_STARTED - 39 seconds, camera stays running')
