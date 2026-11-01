"""Independent sewn Female patterns in the existing armature's local space.

No Male vertex positions are read. Pattern priors are editable visual estimates,
not inferred anatomy or simulated cloth. Open neck/cuffs/hems/slits are intended.
"""
import math
import bpy
import bmesh
import numpy as np
from mathutils import Matrix

CATEGORIES={'FEMALE_KAMEEZ':('Female Kameez','Tops'),
            'FEMALE_TROUSER':('Female Pajama / straight trouser','Bottoms'),
            'FEMALE_SHALWAR':('Female Shalwar / gathered','Bottoms'),
            'FEMALE_DUPATTA':('Separate Female Dupatta','Dupatta')}
NAMES={'FEMALE_KAMEEZ':'MASTER_FEMALE_KAMEEZ',
       'FEMALE_TROUSER':'MASTER_FEMALE_PAJAMA_TROUSER',
       'FEMALE_SHALWAR':'MASTER_FEMALE_SHALWAR',
       'FEMALE_DUPATTA':'MASTER_FEMALE_DUPATTA'}
CONTROLS=('shoulder_width','chest_width','waist_width','hip_width','torso_length',
          'torso_depth','chest_depth','waist_depth','hip_depth','arms','legs')
STYLES=('length','sleeve_length','sleeve_width','hem_width','slit_height','neckline')
VERSION=1

def smooth(v):
    v=np.clip(v,0,1);return v*v*(3-2*v)

class Pattern:
    def __init__(self):self.v=[];self.f=[];self.w=[];self.reg=[]
    def vertex(self,co,weights):
        self.v.append(co);self.w.append(weights);return len(self.v)-1
    def face(self,ids,region=0):self.f.append(tuple(ids));self.reg.append(region)
    def bridge(self,a,b,region=0,skip=()):
        for i in range(len(a)):
            if i not in skip:self.face((a[i],a[(i+1)%len(a)],b[(i+1)%len(a)],b[i]),region)

def upper(body=False):
    p=Pattern();n=64;rings=[]
    levels=([(188,41,24),(201,43,26),(219,34,22),(244,29,18),
             (267,36,23),(282,39,25),(295,38,20),(308,33,16),(316,11,10)] if body else
            [(96,49,28),(99,49,28),(123,48,28),(150,47,28),(176,46,28),
             (199,46,29),(219,42,27),(244,38,24),(267,40,28),
             (282,43,29),(295,41,23),(308,35,18),(316,13,12)])
    for k,(y,rx,rz) in enumerate(levels):
        ring=[]
        for i in range(n):
            a=2*math.pi*i/n;x=rx*math.cos(a);z=-5+rz*math.sin(a)
            yy=y
            # Open round front neck with shallow back; no cylindrical male band.
            if k==len(levels)-1:yy-=10*max(0,math.sin(a))**1.5+2*max(0,-math.sin(a))
            if not body and y<219:
                z+=.65*float(smooth((219-y)/110))*math.cos(8*a)*math.sin(a)
            if y<244:
                spine=.35*float(smooth((y-210)/34));leg=.10*float(smooth((199-y)/90))
                left=float(smooth((x+18)/36))
                weights={'Hips':1-spine-leg,'Spine':spine,'LeftUpLeg':leg*left,'RightUpLeg':leg*(1-left)}
            elif y<282:
                t=float(smooth((y-244)/38));weights={'Spine':1-t,'Spine2':t}
            else:
                t=.6*float(smooth((abs(x)-25)/14));weights={'Spine2':1-t,('LeftArm' if x>=0 else 'RightArm'):t}
            ring.append(p.vertex((x,yy,z),weights))
        rings.append(ring)
    hole=set(range(60,64))|set(range(4))|set(range(28,36))
    for k in range(len(levels)-1):
        y=levels[k][0];skip=hole if y in (282,295) else set()
        if not body and y<199:skip=skip|{0,31,32,63}
        p.bridge(rings[k],rings[k+1],skip=skip)
    edges={}
    for f in p.f:
        for a,b in zip(f,f[1:]+f[:1]):
            key=tuple(sorted((a,b)));edges[key]=edges.get(key,0)+1
    for sign,side,region in [(1,'Left',1),(-1,'Right',2)]:
        boundary=[e for e,v in edges.items() if v==1 and all(281<p.v[i][1]<309 and sign*p.v[i][0]>24 for i in e)]
        adjacency={}
        for a,b in boundary:adjacency.setdefault(a,[]).append(b);adjacency.setdefault(b,[]).append(a)
        if not adjacency or any(len(v)!=2 for v in adjacency.values()):raise ValueError('Female armhole boundary invalid')
        root=[min(adjacency)];last=None;now=root[0]
        while True:
            nxt=next(v for v in adjacency[now] if v!=last)
            if nxt==root[0]:break
            root.append(nxt);last,now=now,nxt
        co=np.array([p.v[i] for i in root]);angles=np.arctan2(co[:,2]+5,co[:,1]-295)
        previous=root
        for k in range(1,17):
            t=k/16;x=sign*(40+103*t);cy=295+5*min(1,t*3)
            radius=(13*(1-t)+6.5*t) if body else (17*(1-t)+11*t)
            lower=float(smooth((abs(x)-83)/22));ring=[]
            for a in angles:ring.append(p.vertex((x,cy+radius*math.cos(a),-10+radius*math.sin(a)),{side+'Arm':1-lower,side+'ForeArm':lower}))
            p.bridge(previous,ring,region);previous=ring
    return p

def lower(gathered=False,body=False):
    p=Pattern();ring=None;n=64
    # Female waist/seat independently drafted; a shared saddle closes crotch.
    specs=([(213,29,20),(209,30,21),(198,39,26),(185,43,28),(176,41,25)] if body else
           [(213,32,21),(209,33,22),(198,39,24),(185,41,25),(173,41,24)])
    for y,rx,rz in specs:
        nxt=[]
        for i in range(n):
            a=-math.pi/2+2*math.pi*i/n
            fold=1+(0 if body else .016)*math.cos(16*a)*float(smooth((213-y)/18))
            x=rx*math.cos(a)*fold;z=-5+rz*math.sin(a)*fold
            t=.32*float(smooth((213-y)/40));left=float(smooth((x+16)/32))
            nxt.append(p.vertex((x,y,z),{'Hips':1-t,'LeftUpLeg':t*left,'RightUpLeg':t*(1-left)}))
        if ring:p.bridge(ring,nxt)
        ring=nxt
    saddle=[ring[32]];end_y=specs[-1][0];rz=specs[-1][2]
    for i in range(1,8):
        t=i/8;saddle.append(p.vertex((0,end_y-8*math.sin(math.pi*t),-5+rz*(1-2*t)),{'Hips':.7,'LeftUpLeg':.15,'RightUpLeg':.15}))
    saddle.append(ring[0])
    angles=[-math.pi/2+math.pi*i/32 for i in range(33)]+[math.pi/2+math.pi*i/8 for i in range(1,8)]
    for sign,side,arc in [(1,'Left',ring[:33]),(-1,'Right',[ring[0]]+list(reversed(ring[32:])) )]:
        previous=arc+saddle[1:-1];centre=18.5*sign
        for y in [158,145,129,119,110,101,88,73,58,43,32,29]:
            radius=float(np.interp(y,[29,73,110,158],[7.5 if gathered else 11,17 if gathered else 12,23 if gathered else 15,25 if gathered else 21]))
            if body:radius=float(np.interp(y,[29,73,110,158],[6,9,12,20]))
            cy=-5;ring2=[];blend=float(smooth((126-y)/32));pelvis=.08*float(smooth((y-140)/18))
            for a in angles:
                ring2.append(p.vertex((centre+sign*radius*math.cos(a),y,cy+radius*1.08*math.sin(a)),
                    {'Hips':pelvis,side+'UpLeg':(1-pelvis)*(1-blend),side+'Leg':(1-pelvis)*blend}))
            p.bridge(previous,ring2);previous=ring2
    return p

def make(name,pattern,collection):
    obj=bpy.data.objects.get(name)
    if obj:return obj
    mesh=bpy.data.meshes.new(name+' Mesh');mesh.from_pydata(pattern.v,[],pattern.f);mesh.update()
    obj=bpy.data.objects.new(name,mesh);collection.objects.link(obj)
    obj.parent=bpy.data.objects['Armature'];obj.matrix_parent_inverse=Matrix.Identity(4);obj.matrix_basis=Matrix.Identity(4)
    obj.modifiers.new('Shared armature','ARMATURE').object=obj.parent
    # Stable complete bone schema, independent of Male mesh group order.
    for bone in obj.parent.data.bones:obj.vertex_groups.new(name=bone.name)
    for i,weights in enumerate(pattern.w):
        total=sum(weights.values())
        for name,w in weights.items():
            if w>1e-8:obj.vertex_groups['mixamorig:'+name].add([i],w/total,'REPLACE')
    region=mesh.attributes.new('GS_pattern_region','INT','FACE');region.data.foreach_set('value',pattern.reg)
    bm=bmesh.new();bm.from_mesh(mesh)
    # Fixed master refinement, once at authoring time. Interpolates deform
    # weights and face regions; every photo variant reuses this same topology.
    bmesh.ops.subdivide_edges(bm,edges=list(bm.edges),cuts=1,use_grid_fill=True,smooth=.30)
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(mesh);bm.free()
    for f in mesh.polygons:f.use_smooth=True
    mesh['gs_female_master_version']=VERSION;obj['gs_profile']='FEMALE';obj['gs_foundation']='independent_female_pattern_v1'
    obj.hide_viewport=True;obj.hide_render=True
    return obj

def ensure(category,collection):
    name=NAMES[category];old=bpy.data.objects.get(name)
    if old:return old
    if category=='FEMALE_DUPATTA':
        import mint_outfit
        return mint_outfit.create(name,mint_outfit.dupatta(),category,collection)
    obj=make(name,upper() if category=='FEMALE_KAMEEZ' else lower(category=='FEMALE_SHALWAR'),collection)
    obj['gs_category']=category;obj['gs_master']=True
    obj['gs_side_vents']=category=='FEMALE_KAMEEZ'
    obj['gs_crotch']='shared saddle' if category!='FEMALE_KAMEEZ' else 'not applicable'
    return obj

def proxy(collection):
    old=bpy.data.objects.get('FEMALE_BODY_PROXY')
    if old:return old
    p=upper(True);q=lower(body=True);offset=len(p.v)
    p.v+=q.v;p.w+=q.w;p.f += [tuple(i+offset for i in f) for f in q.f];p.reg+=q.reg
    obj=make('FEMALE_BODY_PROXY',p,collection);regional_keys(obj);obj['gs_internal_proxy']=True
    return obj

def regional_keys(obj):
    if not obj.data.shape_keys:obj.shape_key_add(name='Basis')
    co=np.array([v.co[:] for v in obj.data.shape_keys.key_blocks[0].data]);x,y,z=co.T
    torso=1-smooth((np.abs(x)-30)/18);arm=smooth((np.abs(x)-34)/20)*smooth((y-267)/16)
    chest=np.exp(-((y-277)/23)**2)*torso;waist=np.exp(-((y-244)/24)**2)*torso
    hip=np.exp(-((y-196)/29)**2)*(1-arm);shoulder=smooth((y-274)/30)*(1-arm)
    for control in CONTROLS:
        key=obj.data.shape_keys.key_blocks.get('Female '+control) or obj.shape_key_add(name='Female '+control)
        dest=co.copy()
        if control.endswith('_width'):
            weight={'shoulder_width':shoulder,'chest_width':chest,'waist_width':waist,'hip_width':hip}[control]
            dest[:,0]+=x*.25*weight
        elif control.endswith('_depth'):
            weight={'torso_depth':torso*smooth((y-188)/20),'chest_depth':chest,'waist_depth':waist,'hip_depth':hip}[control]
            dest[:,2]+=(z+5)*.25*weight
        elif control=='torso_length':dest[:,1]+=(y-209)*.18*smooth((y-209)/35)
        elif control=='arms':dest[:,0]+=np.sign(x)*np.maximum(abs(x)-40,0)*.20*arm
        elif control=='legs':dest[:,1]-=np.maximum(195-y,0)*.15
        key.data.foreach_set('co',dest.ravel());key.slider_min=-1;key.slider_max=1
    obj['gs_regional_controls']='visual proportions; not circumference measurements'
    if obj.get('gs_category')=='FEMALE_KAMEEZ':
        for control in STYLES:
            key=obj.data.shape_keys.key_blocks.get('Style '+control) or obj.shape_key_add(name='Style '+control)
            dest=co.copy();below=np.clip((230-y)/max(230-y.min(),1),0,1)
            if control=='length':dest[:,1]-=28*below
            elif control=='sleeve_length':dest[:,0]+=np.sign(x)*np.maximum(abs(x)-40,0)*.25*arm
            elif control=='sleeve_width':
                dest[:,1]+=(y-300)*.30*arm;dest[:,2]+=(z+10)*.30*arm
            elif control=='hem_width':dest[:,0]+=x*.22*below
            elif control=='slit_height':dest[:,1]+=18*np.exp(-((y-199)/26)**2)*(1-arm)
            elif control=='neckline':dest[:,1]-=12*np.exp(-(x/8)**2)*smooth((y-290)/15)*smooth((z+2)/12)
            key.data.foreach_set('co',dest.ravel());key.slider_min=-1;key.slider_max=1

def apply_regions(obj,values):
    if not obj or not obj.data.shape_keys:return
    for name in CONTROLS:
        key=obj.data.shape_keys.key_blocks.get('Female '+name)
        if key:key.value=float(values.get(name,0))
