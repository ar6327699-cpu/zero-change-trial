import sys,json,math,time
sys.dont_write_bytecode=True
from pathlib import Path
import bpy,numpy as np
from mathutils import Matrix
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'FahadAR'))
import garment_fit
from female_pose import control_heads
BASE=Matrix.Rotation(math.pi/2,4,'X')@Matrix.Diagonal((.01,.01,.01,1.))
arm=bpy.data.objects['Armature'];top=bpy.data.objects['Tops'];arm.data.pose_position='POSE'
base=np.array([tuple(v.co) for v in top.data.vertices]);uv=np.array([tuple(v.uv) for v in top.data.uv_layers.active.data])
fitter=garment_fit.GarmentFitter(arm);surface=fitter.coverage
for m in top.modifiers:
 if m.type=='SUBSURF':m.show_viewport=False
fit=np.array([.45,.45,0,-.95]);arm.matrix_world=Matrix.Translation((0,0,fit[3]))@Matrix.Diagonal((fit[0],fit[0],fit[1],1.))@BASE
_,mod,sockets=fitter.items['Tops']
def evaluated():
 bpy.context.view_layer.update();obj=top.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=obj.to_mesh()
 a=np.array([tuple(obj.matrix_world@v.co) for v in mesh.vertices])
 index=np.array([v.value for v in mesh.attributes['AR_source_index'].data]);assert len(set(index))==len(base)
 a=a[np.argsort(index)];obj.to_mesh_clear();return a
checks=[];metrics=[];samples=[]
for yaw,angle in [(0,-85),(0,70),(65,-85),(-65,-85),(90,-85),(-90,70)]:
 for b in arm.pose.bones:b.matrix_basis=Matrix.Identity(4)
 for side,sgn in [('Left',1),('Right',-1)]:
  b=arm.pose.bones['mixamorig:'+side+'Arm'];head=b.bone.head_local
  b.matrix=Matrix.Translation(head)@Matrix.Rotation(math.radians(angle*sgn),4,'Z')@Matrix.Translation(-head)@b.bone.matrix_local
 bpy.context.view_layer.update()
 # Root rotation carries existing local arm articulation.
 hips=arm.pose.bones['mixamorig:Hips'];hips.matrix=Matrix.Rotation(math.radians(yaw),4,'Y')@hips.bone.matrix_local
 bpy.context.view_layer.update()
 source=control_heads(arm,BASE,garment_fit.CONTROL_BONES)
 p=np.zeros((33,4));p[:,:2]=.5;p[:,3]=.99
 for i,v in source.items():
  screen=v*np.array([fit[0],fit[0],fit[1]])+[0,0,fit[3]]
  p[i,:2]=[screen[0]/(32/9)+.5,.5-screen[2]/2]
 p[[13,14],0]+=[.003,-.003];p[[15,16],1]+=.009
 fitter.reset();fitter.update(p,source,fit,1.)
 predicted=surface.predicted(sockets,mod)[0];before=evaluated()
 prediction_error=float(np.max(np.linalg.norm(predicted-before,axis=1))*180)
 assert prediction_error<.01,(yaw,angle,prediction_error)
 checks.append(f'Native cage prediction matches at yaw {yaw}, arm {angle}')
 screen=np.c_[before[:,0]*180+320,180-before[:,2]*180]
 body=[(float(np.median(screen[ids,1])),float(min(screen[ids,0])-4),float(max(screen[ids,0])+6),ids) for ids in surface.body]
 body.sort(key=lambda r:r[0]);rows=[list(r[:3]) for r in body]
 evidence={'stamp':1.,'torso_px':75.,'rows':rows,'sleeves':{}}
 started=time.perf_counter();surface.update(evidence,1.,p,sockets,mod);elapsed=(time.perf_counter()-started)*1000
 after=evaluated();out=np.c_[after[:,0]*180+320,180-after[:,2]*180]
 errors=[]
 for y,lo,hi,ids in body:
  errors.extend([abs(min(out[ids,0])-lo),abs(max(out[ids,0])-hi)])
 assert max(errors)<.02,(yaw,angle,max(errors))
 checks.append(f'Observed body edges covered at yaw {yaw}, arm {angle}')
 repeat=after.copy();surface.update(evidence,1.,p,sockets,mod);assert np.max(abs(evaluated()-repeat))<1e-6
 checks.append(f'Stationary coverage does not drift at yaw {yaw}, arm {angle}')
 surface.update(None,1.2,p,sockets,mod);assert np.max(abs(evaluated()-before))<1e-6
 checks.append(f'Missing outline clears offsets at yaw {yaw}, arm {angle}')
 metrics.append({'yaw':yaw,'angle':angle,'cage_prediction_error_px':prediction_error,'outline_error_px':max(errors),'update_ms':elapsed})
 samples.append({'yaw':yaw,'angle':angle,'before':screen.tolist(),'after':out.tolist(),'rows':rows})
assert np.array_equal(base,np.array([tuple(v.co) for v in top.data.vertices]))
assert np.array_equal(uv,np.array([tuple(v.uv) for v in top.data.uv_layers.active.data]))
checks.extend(['Authored vertices unchanged','UV coordinates unchanged'])
# Inspect native topology, not only point bounds: the side vents must be
# lined through every lower ring while neck, cuffs and hem stay open.
surface.mod.show_viewport=True;bpy.context.view_layer.update()
obj=top.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=obj.to_mesh()
source_ids=np.array([v.value for v in mesh.attributes['AR_source_index'].data]);edges={}
for face in mesh.polygons:
 ids=list(face.vertices)
 for a,b in zip(ids,ids[1:]+ids[:1]):
  key=tuple(sorted((int(source_ids[a]),int(source_ids[b]))));edges[key]=edges.get(key,0)+1
holes=[(a,b) for (a,b),count in edges.items() if count==1 and abs(base[a,1]-base[b,1])>1 and
       all(abs(base[i,0])>30 and base[i,1]<=200 for i in (a,b))]
assert not holes,holes
assert len(mesh.polygons)==len(top.data.polygons)+len(surface.lining.data.polygons)
obj.to_mesh_clear();checks.append('Native live mesh closes both side vents without editing source topology')
report={'checks_passed':len(checks),'checks':checks,'metrics':metrics,'scene_saved':False}
Path(__file__).with_name('surface_validation.json').write_text(json.dumps(report,indent=2))
Path(__file__).with_name('surface_preview_data.json').write_text(json.dumps({'samples':samples,'faces':[list(p.vertices) for p in top.data.polygons]}))
print('SURFACE_VALIDATION',json.dumps(report))
