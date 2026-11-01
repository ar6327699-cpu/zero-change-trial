"""Read-only blend smoke/deformation check. Never save the scene or start camera."""
import sys
sys.dont_write_bytecode=True
import json,time,hashlib
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'FahadAR'))
import ar_runtime
from female_arm_stabilizer import ArmStabilizer
arm=bpy.data.objects['Armature'];top=bpy.data.objects['Tops']
assert Path(bpy.data.filepath).resolve()==ROOT/'fahadGenius_AR_experiment.blend'
assert top.get('gs_category')=='FEMALE_KAMEEZ',top.get('gs_category')
base=np.array([tuple(v.co) for v in top.data.vertices]);uv=np.array([tuple(v.uv) for v in top.data.uv_layers.active.data])
original={b.name:b.matrix_basis.copy() for b in arm.pose.bones}
arm.data.pose_position='POSE'
def evaluated():
    started=time.perf_counter();bpy.context.view_layer.update()
    obj=top.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=obj.to_mesh()
    a=np.empty(len(mesh.vertices)*3,dtype=np.float32);mesh.vertices.foreach_get('co',a);obj.to_mesh_clear()
    return a.reshape(-1,3),(time.perf_counter()-started)*1000
neutral,ms=evaluated();states={};timings=[]
for side in ['Left','Right']:
    for degrees in [-35,0,35]:
        for n,m in original.items():arm.pose.bones[n].matrix_basis=m
        bone=arm.pose.bones['mixamorig:'+side+'Arm']
        bone.matrix_basis=original[bone.name]@Matrix.Rotation(np.deg2rad(degrees),4,'Z')
        a,ms=evaluated();assert a.shape==neutral.shape and np.isfinite(a).all()
        states[f'{side}_{degrees}']=float(np.max(np.linalg.norm(a-neutral,axis=1)));timings.append(ms)
for n,m in original.items():arm.pose.bones[n].matrix_basis=m
restored,_=evaluated()
assert np.allclose(neutral,restored,atol=1e-4)
assert np.array_equal(base,np.array([tuple(v.co) for v in top.data.vertices]))
assert np.array_equal(uv,np.array([tuple(v.uv) for v in top.data.uv_layers.active.data]))
assert all(v>.01 for k,v in states.items() if not k.endswith('_0'))
report=dict(blend=str(ROOT/'fahadGenius_AR_experiment.blend'),category=top.get('gs_category'),
            runtime_import=True,finite_deformation=True,base_mesh_unchanged=True,uv_unchanged=True,
            sleeve_displacement_blender_units=states,evaluation_ms=timings,
            camera_started=False,scene_saved=False,
            note='Headless authored-rig deformation smoke test; not a live wearer alignment or display-latency test.')
Path(__file__).with_name('blender_validation.json').write_text(json.dumps(report,indent=2))
print('SLEEVE_BLENDER_CHECK '+json.dumps(report))
