import sys,json,math
sys.dont_write_bytecode=True
from pathlib import Path
import bpy,numpy as np
from mathutils import Matrix,Vector
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'FahadAR'))
import garment_fit
arm=bpy.data.objects['Armature'];top=bpy.data.objects['Tops']
base=np.array([tuple(v.co) for v in top.data.vertices])
report={'top':dict(top.items()),'modifiers':[(m.name,m.type,m.show_viewport) for m in top.modifiers],
 'top_matrix':list(map(list,top.matrix_world)), 'parent_inverse':list(map(list,top.matrix_parent_inverse)),
 'shapes':[(k.name,k.value) for k in top.data.shape_keys.key_blocks],
 'bones':{n:{'head':list(arm.data.bones['mixamorig:'+n].head_local),'tail':list(arm.data.bones['mixamorig:'+n].tail_local)} for n in garment_fit.CONTROL_BONES.values()}}
for bone in arm.pose.bones:bone.matrix_basis=Matrix.Identity(4)
arm.data.pose_position='POSE';arm.matrix_world=Matrix.Identity(4)
fitter=garment_fit.GarmentFitter(arm)
def evaluated():
 bpy.context.view_layer.update()
 obj=top.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=obj.to_mesh()
 a=np.array([tuple(obj.matrix_world@v.co) for v in mesh.vertices]);obj.to_mesh_clear();return a
rest=evaluated();report['vertices']=[len(base),len(rest)]
report['rings']={}
for side,sgn in [('Left',1),('Right',-1)]:
 xs=np.unique(np.round(base[(base[:,0]*sgn>42)&(base[:,1]>275),0],3))
 for x in xs[::max(1,len(xs)//5)]:
  ids=np.flatnonzero((abs(base[:,0]-x)<.005)&(base[:,1]>275))
  if len(ids)<5:continue
  report['rings'][str(x)]={'count':len(ids),'base_center':base[ids].mean(0).tolist(),'rest_center':rest[ids].mean(0).tolist(),
   'fit_weights':{str(j):float(np.mean([top.data.attributes['AR_fit_'+str(j)].data[i].value for i in ids])) for j in fitter.items['Tops'][2]},
   'skin_weights':{g.name:float(np.mean([sum(q.weight for q in top.data.vertices[i].groups if q.group==g.index) for i in ids])) for g in top.vertex_groups if any(k in g.name for k in ['Arm','Hand','Shoulder'])}}
report['fit_modifiers_after']=[(m.name,m.type,m.show_viewport) for m in top.modifiers]
Path(__file__).with_name('rig_inspection.json').write_text(json.dumps(report,indent=2,default=str))
print('INSPECT_RIG',json.dumps({k:report[k] for k in ['modifiers','shapes','vertices','rings','fit_modifiers_after']}))
