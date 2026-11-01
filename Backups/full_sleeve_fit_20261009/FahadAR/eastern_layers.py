"""Paired, deterministic garment ease; never a nearest-surface shrinkwrap.

The covered seat is inside the upper, in the same pelvis frame. Leg bending
blends in below the upper hem, preventing the lower's gathered waistband and
knees from piercing a straight hanging kameez. Only runtime copies are edited.
"""
import hashlib
import numpy as np

VERSION=3
UPPERS={'KURTA','FEMALE_KAMEEZ'}
LOWERS={'SHALWAR','PAJAMA','FEMALE_TROUSER','FEMALE_SHALWAR'}

def ease(t):
    t=np.clip(t,0,1);return t*t*(3-2*t)

def basis(obj):
    return np.array([v.co[:] for v in (obj.data.shape_keys.key_blocks[0].data if obj.data.shape_keys else obj.data.vertices)],float)

def restore_weights(obj):
    groups=[g for g in obj.vertex_groups if g.name.startswith('mixamorig:')]
    result=np.zeros((len(obj.data.vertices),len(groups)))
    for j,g in enumerate(groups):
        name='GS_layer_original_'+g.name.split(':')[-1]
        attr=obj.data.attributes.get(name)
        if attr is None:
            values=np.array([next((r.weight for r in v.groups if r.group==g.index),0.) for v in obj.data.vertices],np.float32)
            attr=obj.data.attributes.new(name,'FLOAT','POINT');attr.data.foreach_set('value',values)
        else:
            values=np.empty(len(obj.data.vertices),np.float32);attr.data.foreach_get('value',values)
        result[:,j]=values
    return groups,result

def weight_to_pelvis(obj,influence):
    groups,w=restore_weights(obj);indices=list(range(len(w)))
    hip=next(i for i,g in enumerate(groups) if g.name=='mixamorig:Hips')
    w*=1-influence[:,None];w[:,hip]+=influence
    for j,g in enumerate(groups):
        g.remove(indices)
        for i in np.flatnonzero(w[:,j]>1e-7):g.add([int(i)],float(w[i,j]),'REPLACE')

def paired_weights(obj,coords,hem,is_top):
    groups,w=restore_weights(obj);indices=list(range(len(w)));column={g.name.split(':')[-1]:i for i,g in enumerate(groups)}
    y=coords[:,1];side=ease((coords[:,0]+34)/68)
    thigh=.90*ease((211-y)/max(211-hem,25))
    target=np.zeros_like(w);target[:,column['Hips']]=1-thigh
    target[:,column['LeftUpLeg']]=thigh*side;target[:,column['RightUpLeg']]=thigh*(1-side)
    amount=1-ease((y-211)/30)
    if not is_top:amount*=ease((y-hem+32)/32)
    w=w*(1-amount[:,None])+target*amount[:,None]
    for j,g in enumerate(groups):
        g.remove(indices)
        for i in np.flatnonzero(w[:,j]>1e-7):g.add([int(i)],float(w[i,j]),'REPLACE')

def ensure(top,bottom):
    if top.get('gs_category') not in UPPERS or bottom.get('gs_category') not in LOWERS:return
    c=basis(top);b=basis(bottom);hem=float(c[:,1].min())
    female=top.get('gs_category')=='FEMALE_KAMEEZ'
    import male_eastern_finish
    male=male_eastern_finish.owned(top) and male_eastern_finish.owned(bottom)
    signature=hashlib.sha256(c.astype(np.float32).tobytes()+str(bottom.get('gs_id','')).encode()+str('female4' if female else 'male5' if male else VERSION).encode()).hexdigest()
    if bottom.data.get('GS_layer_pair')==signature and top.data.get('GS_layer_pair')==signature:return
    # Estimate ring envelope only from torso, below the shoulders. Group nearby
    # subdivided samples into narrow horizontal slices; all assets share units.
    levels=[]
    for y in np.arange(hem,248,3.):
        near=c[abs(c[:,1]-y)<5]
        if len(near)<8:
            nearest=abs(c[:,1]-y);near=c[nearest<=np.partition(nearest, min(63,len(c)-1))[min(63,len(c)-1)]]
        centre=np.array([(near[:,0].min()+near[:,0].max())*.5,(near[:,2].min()+near[:,2].max())*.5])
        radii=np.ptp(near[:,[0,2]],axis=0)*.5
        levels.append((y,*centre,*np.maximum(radii-3.5,3)))
    levels=np.array(levels);y=b[:,1]
    cx=np.interp(y,levels[:,0],levels[:,1]);cz=np.interp(y,levels[:,0],levels[:,2])
    rx=np.interp(y,levels[:,0],levels[:,3]);rz=np.interp(y,levels[:,0],levels[:,4])
    radius=np.sqrt(((b[:,0]-cx)/rx)**2+((b[:,2]-cz)/rz)**2)
    factor=np.minimum(1.,1/np.maximum(radius,1e-6))
    covered=ease((y-hem+14)/14)
    factor=1+(factor-1)*covered
    fitted=b.copy();fitted[:,0]=cx+(b[:,0]-cx)*factor;fitted[:,2]=cz+(b[:,2]-cz)*factor
    if not bottom.data.shape_keys:bottom.shape_key_add(name='Basis')
    key=bottom.data.shape_keys.key_blocks.get('Eastern paired ease') or bottom.shape_key_add(name='Eastern paired ease')
    key.data.foreach_set('co',fitted.ravel());key.value=1.
    # Straight upper panels hang from the pelvis, not separate knee estimates.
    if female or male:
        # A slit kameez hangs from the pelvis. Blending both independent thigh
        # rotations 90% into its hem fans the two panels during turns/walking.
        weight_to_pelvis(top,1-ease((c[:,1]-211)/30))
    else:paired_weights(top,c,hem,True)
    # Covered part and a short transition below hem share that frame. Calves
    # still follow authored leg weights; this prioritizes clean stable layers.
    if male:
        # Covered fabric and upper hem share a pelvis frame at every yaw.
        # Release independent leg bending smoothly below the hem, rather
        # than pushing intersecting fabric along a camera-dependent ray.
        weight_to_pelvis(bottom,ease((b[:,1]-hem+28)/28))
    elif female:
        # Restore the authored continuous seat and independent legs. Do not
        # make the trouser follow the kameez's skirt instead of its own knees.
        weight_to_pelvis(bottom,np.zeros(len(b)))
    else:paired_weights(bottom,b,hem,False)
    bottom.data['GS_layer_pair']=signature;top.data['GS_layer_pair']=signature;bottom['gs_layer_hem']=hem
    top.update_tag();bottom.update_tag()
    return {'hem':hem,'inset_vertices':int(np.count_nonzero(factor<.999)),'version':VERSION}
