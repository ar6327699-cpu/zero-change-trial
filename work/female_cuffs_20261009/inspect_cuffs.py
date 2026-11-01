import bpy,json,sys
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
out=Path(__file__).resolve().parent
rows=[]
arm=bpy.data.objects['Armature']
for name in ['Tops','GS Profile FEMALE Tops']:
    obj=bpy.data.objects.get(name)
    if not obj:continue
    keys=obj.data.shape_keys
    co=np.array([v.co[:] for v in keys.key_blocks[0].data]) if keys else np.array([v.co[:] for v in obj.data.vertices])
    row=dict(name=name,category=obj.get('gs_category'),label=obj.get('gs_label'),id=obj.get('gs_id'),mesh=obj.data.name,
        keys={k.name:k.value for k in keys.key_blocks} if keys else {},bounds=[co.min(0).tolist(),co.max(0).tolist()],cuffs={})
    for sign,side in [(1,'Left'),(-1,'Right')]:
        select=(co[:,0]*sign>80)&(co[:,1]>270);extreme=float((co[select,0]*sign).max());ring=select&(co[:,0]*sign>extreme-.05)
        row['cuffs'][side]=dict(extreme=extreme,centre=co[ring].mean(0).tolist(),radius_y=float(np.ptp(co[ring,1])/2),radius_z=float(np.ptp(co[ring,2])/2),vertices=int(ring.sum()),wrist=tuple(arm.data.bones['mixamorig:'+side+'Hand'].head_local))
    rows.append(row)
scenes=[dict(name=s.name,settings={k:s[k] for k in s.keys() if k.startswith(('gs_female_style','GS_active_profile','GS_body_profiles'))}) for s in bpy.data.scenes]
report=dict(rows=rows,scenes=scenes,library_count=len([o for o in bpy.data.objects if o.get('gs_category')=='FEMALE_KAMEEZ']))
(out/'inspection.json').write_text(json.dumps(report,indent=2));print('CUFF_INSPECTION '+json.dumps(report))
