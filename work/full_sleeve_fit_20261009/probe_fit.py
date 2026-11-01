import sys,json,math
sys.dont_write_bytecode=True
from pathlib import Path
import bpy,numpy as np
from mathutils import Matrix,Vector
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'FahadAR'))
import garment_fit
from female_pose import control_heads
BASE=Matrix.Rotation(math.pi/2,4,'X')@Matrix.Diagonal((.01,.01,.01,1.))
arm=bpy.data.objects['Armature'];top=bpy.data.objects['Tops'];arm.data.pose_position='POSE'
for b in arm.pose.bones:b.matrix_basis=Matrix.Identity(4)
base=np.array([tuple(v.co) for v in top.data.vertices])
for m in top.modifiers:
 if m.type=='SUBSURF':m.show_viewport=False
fitter=garment_fit.GarmentFitter(arm);fit=np.array([.45,.45,0,-.95])
arm.matrix_world=Matrix.Translation((0,0,fit[3]))@Matrix.Diagonal((fit[0],fit[0],fit[1],1.))@BASE
def evaluated():
 bpy.context.view_layer.update();obj=top.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=obj.to_mesh()
 a=np.array([tuple(obj.matrix_world@v.co) for v in mesh.vertices]);obj.to_mesh_clear();return a
report=[]
for angle in [-85,0,70,90]:
 for side,sgn in [('Left',1),('Right',-1)]:
  b=arm.pose.bones['mixamorig:'+side+'Arm'];head=b.bone.head_local
  b.matrix=Matrix.Translation(head)@Matrix.Rotation(math.radians(angle*sgn),4,'Z')@Matrix.Translation(-head)@b.bone.matrix_local
 bpy.context.view_layer.update()
 source=control_heads(arm,BASE,garment_fit.CONTROL_BONES)
 p=np.zeros((33,4));p[:,:2]=.5;p[:,3]=.99
 for i,v in source.items():
  screen=v*np.array([fit[0],fit[0],fit[1]])+[0,0,fit[3]]
  p[i,:2]=[screen[0]/(32/9)+.5,.5-screen[2]/2]
 for dy in [0.,-.10]:
  test=p.copy();test[[15,16],1]+=dy;fitter.reset();info=fitter.update(test,source,fit,1.)
  vertices=evaluated();row={'angle':angle,'wrist_delta_y':dy,'info':info,'arms':{}}
  for side,sgn,i in [('Left',1,15),('Right',-1,16)]:
   x=(base[:,0]*sgn).max();ids=np.flatnonzero(abs(base[:,0]*sgn-x)<.01)
   centre=vertices[ids].mean(0);screen=np.array([centre[0]*180+320,180-centre[2]*180])
   target=test[i,:2]*[640,360]
   row['arms'][side]={'target':target.tolist(),'cuff':screen.tolist(),'error_px':float(np.linalg.norm(screen-target)),'confidence':test[i,3]}
  report.append(row)
Path(__file__).with_name('probe_fit_before.json').write_text(json.dumps(report,indent=2))
print('PROBE_FIT',json.dumps(report))
