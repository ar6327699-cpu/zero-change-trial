"""Scoped fit revision for the three 2026-09-27 reference Western sets.

Regular untucked T-shirt, cuffs above elbows, straight full-length trousers.
The additional wearer photos guide proportions only; artwork/UVs stay intact.
"""
import numpy as np

TOPS={'293cd54194fb52199b7bb0bc7937c2b3','caf1a53d13f95ab1bcbddc2bfe9c64ed','24806757ec935c55a5bc5187dc79dbaf'}
BOTTOMS={'4db7cf224351509b8900d48a520a691c','0dd9460da44759fbb17c628122ae99f0'}

def owned(obj):
    return obj is not None and obj.get('gs_id') in TOPS|BOTTOMS

def match_reference_contour(obj):
    """Match the final front-reference's chest/waist/hem silhouette ratios.

    Black front panel: chest ~181 px, waist ~166 px, hem ~192 px.
    Use dimensionless ratios, not that product photo's absolute human size.
    Keep shoulder anchors, 80%-upper-arm sleeve length and full-length pants.
    """
    if obj.get('gs_id') in TOPS:
        co=np.array([v.co[:] for v in obj.data.vertices])
        edges=np.array([e.vertices[:] for e in obj.data.edges]);a=co[edges[:,0]];b=co[edges[:,1]]
        levels=np.array([200.,215.,235.,255.,275.]);widths=[]
        for y in levels:
            mask=(np.minimum(a[:,1],b[:,1])<=y)&(np.maximum(a[:,1],b[:,1])>=y)&(abs(a[:,1]-b[:,1])>1e-6)
            aa=a[mask];bb=b[mask];p=aa+(bb-aa)*((y-aa[:,1])/(bb[:,1]-aa[:,1]))[:,None]
            widths.append((p[:,0].max()-p[:,0].min())/2)
        chest=widths[-1]
        ratios=np.array([192/181,1.01,166/181,.96,1.0])
        factor=np.interp(co[:,1],levels,chest*ratios/np.array(widths))
        factor[co[:,1]>=275]=1.
        co[:,0]*=factor
        obj.data.vertices.foreach_set('co',co.ravel());obj.data.update()
        obj.data['gs_reference_width_ratios']=[1.,166/181,192/181]
    obj.data['gs_western_fit_revision']=3
    obj.data['gs_fit_reference']='Final three-set montage 2026-09-27: chest/waist/hem proportions; original artwork retained'

def revise(obj):
    if not owned(obj) or obj.data.get('gs_western_fit_revision',0)>=3:return False
    if obj.data.shape_keys:raise ValueError('Revise the stored Western source before creating runtime shape keys')
    if obj.data.get('gs_western_fit_revision',0)>=2:
        match_reference_contour(obj);return True
    import bpy
    co=np.array([v.co[:] for v in obj.data.vertices]);original=co.copy()
    rig=bpy.data.objects['Armature']
    if obj.get('gs_id') in TOPS:
        x,y,z=original.T
        # The previous native short-sleeve ends were only ~55% of upper-arm
        # length. Grade each side independently to the same 80% proportion.
        for side,sign in [('Left',1),('Right',-1)]:
            shoulder=rig.data.bones['mixamorig:'+side+'Arm'].head_local
            elbow=rig.data.bones['mixamorig:'+side+'ForeArm'].head_local
            start=abs(shoulder.x);end=abs(elbow.x)
            sleeve=(sign*x>start)&(y>275)
            current=np.max(sign*x[sleeve]);target=start+.80*(end-start)
            co[sleeve,0]=sign*(start+(sign*x[sleeve]-start)*(target-start)/(current-start))
        torso=np.clip((45-abs(x))/10,0,1)*np.clip((300-y)/25,0,1)
        co[:,0]+=x*.13*torso
        # Reference's untucked hem covers the waistband, without a long kurta
        # silhouette. No alterations to shoulder anchors or the wearer rig.
        t=np.clip((255-y)/60,0,1);t=t*t*(3-2*t)
        co[:,1]-=10*t
    else:
        # Grade the original lower leg rings into regular straight legs rather
        # than short/narrow cuffs. Keep the seat/crotch and cargo panels intact.
        from western_collection import cross_sections
        y=original[:,1];low=float(y.min());heights=np.linspace(low+.001,160,260)
        bounds=cross_sections(obj,original,heights)
        for side,sign in [(0,-1),(1,1)]:
            mask=(original[:,0]*sign>0)&(y<150)
            indices=np.clip(np.rint((y[mask]-low)/(160-low)*259).astype(int),0,259)
            lo=bounds[indices,side,0];hi=bounds[indices,side,1]
            centre=(lo+hi)/2;half=np.maximum((hi-lo)/2,1)
            thigh=bounds[int((145-low)/(160-low)*259),side]
            thigh_half=(thigh[1]-thigh[0])/2
            target=thigh_half*np.interp(y[mask],[low,110,150],[.80,.90,1.0])
            blend=np.clip((150-y[mask])/30,0,1)
            factor=1+blend*(np.clip(target/half,.85,1.65)-1)
            co[mask,0]=centre+(original[mask,0]-centre)*factor
        # Full length ends just above the foot joint (no floor/shoe extension).
        ankle=np.mean([rig.data.bones['mixamorig:'+s+'Foot'].head_local.y for s in ('Left','Right')])
        co[:,1]-=max(0.,low-ankle-.5)*np.clip((110-y)/(110-low),0,1)
    obj.data.vertices.foreach_set('co',co.ravel());obj.data.update()
    obj.data['gs_western_fit_revision']=2
    obj.data['gs_fit_reference']='Regular untucked shirt and straight jeans; wearer reference photos 2026-09-27'
    match_reference_contour(obj)
    return True
