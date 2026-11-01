"""Inspect a modest fit change in memory and render it; never save the source."""
import bpy,json,sys,hashlib
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
root=OUT.parents[1];source=root/'fahadGenius_AR_experiment.blend'
assert Path(bpy.data.filepath).resolve()==source.resolve()
top=bpy.data.objects['Tops'];arm=bpy.data.objects['Armature']
assert top.get('gs_category')=='FEMALE_KAMEEZ'
keys=top.data.shape_keys.key_blocks
length=float(keys['Style sleeve_length'].value)-.10
width=float(keys['Style sleeve_width'].value)+.30
assert -1<=length<=1 and -1<=width<=1
base=np.array([v.co[:] for v in keys[0].data])
def shaped(obj):
    keys=obj.data.shape_keys.key_blocks;co=np.array([v.co[:] for v in keys[0].data])
    for k in list(keys)[1:]:
        if not k.mute:co+=(np.array([v.co[:] for v in k.data])-np.array([v.co[:] for v in k.relative_key.data]))*k.value
    return co
before=shaped(top);keys['Style sleeve_length'].value=length;keys['Style sleeve_width'].value=width;after=shaped(top)
torso=(abs(base[:,0])<34)|(base[:,1]<267)
assert np.array_equal(before[torso],after[torso])
stats={}
for sign,side in [(1,'Left'),(-1,'Right')]:
    mask=(sign*base[:,0]>80)&(base[:,1]>270);end=np.max(sign*base[mask,0]);ring=mask&(sign*base[:,0]>end-.05)
    old=before[ring];new=after[ring];wrist=np.array(arm.data.bones['mixamorig:'+side+'Hand'].head_local)
    setback=float(sign*(old[:,0].mean()-new[:,0].mean()))
    ratio=float(np.ptp(new[:,1])/np.ptp(old[:,1]))
    assert 2.5<setback<2.8 and 1.085<ratio<1.095
    assert float(np.max(sign*new[:,0]))<sign*wrist[0]
    stats[side]=dict(setback_model_units=setback,width_ratio=ratio,wrist_x=float(wrist[0]),before_cuff_x=float(old[:,0].mean()),after_cuff_x=float(new[:,0].mean()))
report=dict(style_sleeve_length=length,style_sleeve_width=width,cuffs=stats,torso_unchanged=True,source_saved=False)
(OUT/'preview_validation.json').write_text(json.dumps(report,indent=2))

# Isolated rest-space preview: identical camera/light, actual mesh and shape
# deltas. The blue line marks the authored wrist, not a measured human wrist.
scene=bpy.data.scenes.new('Cuff QA');scene.render.engine='BLENDER_WORKBENCH'
scene.render.resolution_x=1000;scene.render.resolution_y=400;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG';scene.render.film_transparent=False
scene.world=bpy.data.worlds.new('Cuff QA World')
shade=scene.display.shading;shade.light='STUDIO';shade.studiolight_rotate_z=0
shade.color_type='OBJECT';shade.background_type='WORLD';shade.background_color=(.85,.85,.85)
shade.show_shadows=True;shade.show_cavity=True;shade.cavity_type='BOTH'
scene.view_settings.view_transform='Standard'
cam=bpy.data.objects.new('Cuff QA Camera',bpy.data.cameras.new('Cuff QA Camera'))
scene.collection.objects.link(cam);scene.camera=cam;cam.location=(113,300,150);cam.rotation_euler=(0,0,0);cam.data.type='ORTHO';cam.data.ortho_scale=105
mesh=bpy.data.meshes.new('Cuff QA mesh');mesh.from_pydata(before.tolist(),[],[list(p.vertices) for p in top.data.polygons]);mesh.update()
obj=bpy.data.objects.new('Cuff QA Sleeve',mesh);scene.collection.objects.link(obj);obj.color=(.76,.39,.47,1)
for p in mesh.polygons:p.use_smooth=True
# Thin line in front of the cloth, positioned exactly at the left wrist x.
wrist=arm.data.bones['mixamorig:LeftHand'].head_local
line=bpy.data.meshes.new('Wrist line');line.from_pydata([(wrist.x-.12,280,25),(wrist.x+.12,280,25),(wrist.x+.12,320,25),(wrist.x-.12,320,25)],[],[(0,1,2,3)])
marker=bpy.data.objects.new('Authored wrist reference',line);marker.color=(.06,.35,.8,1);scene.collection.objects.link(marker)
for name,co in [('before',before),('after',after)]:
    mesh.vertices.foreach_set('co',co.ravel());mesh.update();scene.render.filepath=str(OUT/(name+'.png'))
    bpy.ops.render.render(write_still=True,scene=scene.name)
print('CUFF_PREVIEW '+json.dumps(report))

