import sys,math,json
sys.dont_write_bytecode=True
from pathlib import Path
import bpy,numpy as np
from mathutils import Matrix
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'FahadAR'))
import garment_fit
from female_pose import control_heads
BASE=Matrix.Rotation(math.pi/2,4,'X')@Matrix.Diagonal((.01,.01,.01,1.))
arm=bpy.data.objects['Armature'];top=bpy.data.objects['Tops'];arm.data.pose_position='POSE'
fitter=garment_fit.GarmentFitter(arm);surface=fitter.coverage;_,mod,sockets=fitter.items['Tops']
fit=np.array([.45,.45,0,-.95]);arm.matrix_world=Matrix.Translation((0,0,fit[3]))@Matrix.Diagonal((fit[0],fit[0],fit[1],1.))@BASE
scene=bpy.data.scenes.new('Coverage QA');scene.render.engine='BLENDER_WORKBENCH';scene.render.resolution_x=1400;scene.render.resolution_y=1200;scene.render.resolution_percentage=100
scene.world=bpy.data.worlds.new('QA World');scene.display.shading.color_type='OBJECT';scene.display.shading.light='STUDIO';scene.display.shading.show_shadows=False
scene.display.shading.background_type='WORLD';scene.display.shading.background_color=(.92,.92,.92);scene.view_settings.view_transform='Standard'
cam=bpy.data.objects.new('QA Camera',bpy.data.cameras.new('QA Camera'));scene.collection.objects.link(cam);scene.camera=cam
cam.location=(1.,-10,0);cam.rotation_euler=(math.pi/2,0,0);cam.data.type='ORTHO';cam.data.ortho_scale=4.1
def clone(label,shift,color):
 bpy.context.view_layer.update();obj=top.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=bpy.data.meshes.new_from_object(obj)
 mesh.transform(obj.matrix_world);view=bpy.data.objects.new(label,mesh);scene.collection.objects.link(view);view.location=shift;view.color=color
for row,(yaw,angle) in enumerate([(0,-85),(0,70),(85,-85),(-85,-85)]):
 for b in arm.pose.bones:b.matrix_basis=Matrix.Identity(4)
 for side,sgn in [('Left',1),('Right',-1)]:
  b=arm.pose.bones['mixamorig:'+side+'Arm'];h=b.bone.head_local;b.matrix=Matrix.Translation(h)@Matrix.Rotation(math.radians(angle*sgn),4,'Z')@Matrix.Translation(-h)@b.bone.matrix_local
 bpy.context.view_layer.update();hips=arm.pose.bones['mixamorig:Hips'];hips.matrix=Matrix.Rotation(math.radians(yaw),4,'Y')@hips.bone.matrix_local
 bpy.context.view_layer.update();source=control_heads(arm,BASE,garment_fit.CONTROL_BONES)
 p=np.zeros((33,4));p[:,:2]=.5;p[:,3]=.99
 for i,v in source.items():
  point=v*np.array([fit[0],fit[0],fit[1]])+[0,0,fit[3]];p[i,:2]=[point[0]/(32/9)+.5,.5-point[2]/2]
 fitter.reset();fitter.update(p,source,fit,1.);predicted=surface.predicted(sockets,mod)[0]
 screen=np.c_[predicted[:,0]*180+320,180-predicted[:,2]*180]
 rows=sorted([[float(np.median(screen[ids,1])),float(min(screen[ids,0])-4),float(max(screen[ids,0])+6)] for ids in surface.body])
 shiftz=1.20 if row<2 else -.45;shiftx=0 if row%2==0 else 2.
 surface.mod.show_viewport=False;clone(f'before {yaw} {angle}',(shiftx,0,shiftz),(.38,.43,.48,1));surface.mod.show_viewport=True
 surface.update({'stamp':1.,'torso_px':75.,'rows':rows,'sleeves':{}},1.,p,sockets,mod)
 clone(f'after {yaw} {angle}',(shiftx+.8,0,shiftz),(.82,.42,.5,1))
 for col in (0,.8):
  verts=[((lo-320)/180,1.5,(180-y)/180) for y,lo,hi in rows]+[((hi-320)/180,1.5,(180-y)/180) for y,lo,hi in rows[::-1]]
  mesh=bpy.data.meshes.new('Observed outline');mesh.from_pydata(verts,[],[list(range(len(verts)))]);obj=bpy.data.objects.new('Observed outline',mesh);scene.collection.objects.link(obj);obj.location=(shiftx+col,0,shiftz);obj.color=(.15,.63,.28,1)
scene.render.image_settings.file_format='PNG';scene.render.filepath=str(Path(__file__).with_name('native_surface_qa.png'))
bpy.ops.render.render(write_still=True,scene=scene.name)
print('NATIVE_SURFACE_QA_RENDERED')
