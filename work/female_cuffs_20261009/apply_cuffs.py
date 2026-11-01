"""Save only the requested female sleeve fit controls in the trial blend."""
import bpy,json,sys,hashlib,copy
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
SOURCE=ROOT/'fahadGenius_AR_experiment.blend';BACKUP=ROOT/'Backups/female_cuffs_20261009'
assert Path(bpy.data.filepath).resolve()==SOURCE.resolve()
expected=json.loads((BACKUP/'before_hash.json').read_text(encoding='utf-8-sig'))['Hash']
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest().upper()==expected,'Source changed since inspection'
assert hashlib.sha256((BACKUP/SOURCE.name).read_bytes()).hexdigest().upper()==expected,'Backup mismatch'
settings=json.loads((OUT/'preview_validation.json').read_text())
scene=next(s for s in bpy.data.scenes if s.get('GS_studio_version'))
profiles=json.loads(scene['GS_body_profiles_v1']);before_profiles=copy.deepcopy(profiles)
top=bpy.data.objects['Tops'];assert top.get('gs_category')=='FEMALE_KAMEEZ'
keys=top.data.shape_keys.key_blocks
assert keys['Style sleeve_length'].value==0 and keys['Style sleeve_width'].value==0
def array_hash(data,attr,width):
    a=np.empty(len(data)*width,dtype=np.float32);data.foreach_get(attr,a);return hashlib.sha256(a.tobytes()).hexdigest()
base_hash=array_hash(top.data.vertices,'co',3)
uv_hash={u.name:array_hash(u.data,'uv',2) for u in top.data.uv_layers}
other_keys={k.name:k.value for k in keys if k.name not in ('Style sleeve_length','Style sleeve_width')}
old_bases={k.name:array_hash(k.data,'co',3) for k in keys}
style=profiles['FEMALE'].setdefault('style',{})
for name in ('sleeve_length','sleeve_width'):
    value=settings['style_'+name];style[name]=value
    scene['gs_female_style_'+name]=value;keys['Style '+name].value=value
    # Normally the female slot shares the active mesh; cover independent saved
    # slots too so returning from the male profile uses the same visual trim.
    saved=bpy.data.objects.get('GS Profile FEMALE Tops')
    if saved and saved.data.shape_keys:
        key=saved.data.shape_keys.key_blocks.get('Style '+name)
        if key:key.value=value
scene['GS_body_profiles_v1']=json.dumps(profiles)
assert profiles['MALE']==before_profiles['MALE']
check=copy.deepcopy(profiles);check['FEMALE']['style']=before_profiles['FEMALE']['style']
assert check==before_profiles
assert array_hash(top.data.vertices,'co',3)==base_hash
assert {u.name:array_hash(u.data,'uv',2) for u in top.data.uv_layers}==uv_hash
assert {k.name:k.value for k in keys if k.name not in ('Style sleeve_length','Style sleeve_width')}==other_keys
assert {k.name:array_hash(k.data,'co',3) for k in keys}==old_bases
# Keep the user's existing .blend1 as well as our explicit rollback copy.
bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(SOURCE),check_existing=False)
report=dict(saved=str(SOURCE),female_style=style,male_profile_unchanged=True,
            mesh_and_uv_unchanged=True,shape_key_geometry_unchanged=True,other_controls_unchanged=True,
            backup=str(BACKUP/SOURCE.name),applies_to='Female profile styles reused on female outfit activation',
            live_wearer_fit_verified=False)
(OUT/'applied.json').write_text(json.dumps(report,indent=2));print('CUFF_APPLIED '+json.dumps(report))
