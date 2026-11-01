"""Male Eastern v1: isolated pattern geometry and fit policy.

Rig-local centimetre-like units of this specific Mixamo template, not wearer
measurements. Open hems/cuffs are intentional. No cloth simulation is claimed.
Western generators and photo projection remain unchanged.
"""
import math
import bpy
import bmesh
import numpy as np

CATEGORIES={'KURTA':('Kurta / long full-sleeve','Tops'),
            'SHALWAR':('Shalwar / gathered loose lower','Bottoms')}
LONG_UPPERS=frozenset({'KURTA'})
LONG_LOWERS=frozenset({'SHALWAR'})
EXTRA_TOP_GROUPS=('mixamorig:LeftUpLeg','mixamorig:RightUpLeg')

def smooth(t):
    t=max(0.,min(1.,t));return t*t*(3-2*t)

def finish(obj):
    bm=bmesh.new();bm.from_mesh(obj.data)
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
    bm.to_mesh(obj.data);bm.free();obj.data.update()
    for p in obj.data.polygons:p.use_smooth=True
    obj['gs_foundation']='male_eastern_v1'
    obj['gs_photo_requirement']='Front/back flat-lay, full garment visible, sleeves spread'

def kurta(obj,extend_sleeves):
    """Retain proven shoulder/arm skinning, replace lower cut with a long skirt.

    Fourteen graded rings cover waist, hip, thigh, and knee-length hem. Side
    openings below the hip separate front/back panels during leg movement.
    Lower-panel skinning deliberately blends pelvis and upper legs, not knees.
    """
    extend_sleeves(obj,'FULL_SHIRT')
    for name in EXTRA_TOP_GROUPS:
        if obj.vertex_groups.get(name) is None:obj.vertex_groups.new(name=name)
    groups={g.name:g.index for g in obj.vertex_groups}
    arm=bpy.data.objects['Armature'];hip=arm.data.bones['mixamorig:Hips'].head_local.y
    knee=arm.data.bones['mixamorig:LeftLeg'].head_local.y
    cut=hip+16;end=knee+8
    bm=bmesh.new();bm.from_mesh(obj.data);deform=bm.verts.layers.deform.verify()
    bmesh.ops.bisect_plane(bm,geom=list(bm.verts)+list(bm.edges)+list(bm.faces),
        dist=.0001,plane_co=(0,cut,0),plane_no=(0,-1,0),clear_outer=True)
    edges=[e for e in bm.edges if e.is_boundary and all(abs(v.co.y-cut)<.002 for v in e.verts)]
    if not edges:bm.free();raise ValueError('Kurta waist cut has no boundary')
    origin={v:v.co.copy() for e in edges for v in e.verts}
    maxx=max(abs(v.co.x) for v in origin)
    maxz=max(abs(v.co.z) for v in origin)
    def ease(t,extent,target):
        # Seat ease is reached before the widest Shalwar hip ring, not only
        # at the hem. A long garment must enclose its lower layer naturally.
        return 1+(max(1,target/extent)-1)*smooth(t/.38)+.035*t
    for step in range(1,15):
        t=step/14;prev=(step-1)/14
        result=bmesh.ops.extrude_edge_only(bm,edges=edges)
        new={v for v in result['geom'] if isinstance(v,bmesh.types.BMVert)}
        for v in new:
            v.co.y=cut+(end-cut)*t
            v.co.x*=ease(t,maxx,54)/ease(prev,maxx,54)
            v.co.z*=ease(t,maxz,39)/ease(prev,maxz,39)
            leg=.85*smooth((hip-v.co.y)/max(hip-end,1))
            left=smooth((v.co.x+12)/24)
            spine=.30*(1-smooth((cut-v.co.y)/24))
            d=v[deform];d.clear()
            for name,w in [('Hips',1-leg-spine),('Spine',spine),('LeftUpLeg',leg*left),('RightUpLeg',leg*(1-left))]:
                if w>1e-6:d[groups['mixamorig:'+name]]=w
        # Small side vents, no welded skirt bridge across the lateral thighs.
        if cut+(end-cut)*prev < hip-13:
            remove=[f for f in result['geom'] if isinstance(f,bmesh.types.BMFace)
                    and all(abs(v.co.x)>maxx*ease(prev,maxx,54)*.97 for v in f.verts)]
            if remove:bmesh.ops.delete(bm,geom=remove,context='FACES_ONLY')
        edges=[e for e in result['geom'] if isinstance(e,bmesh.types.BMEdge) and e.is_valid and all(v in new for v in e.verts)]
    bm.to_mesh(obj.data);bm.free();finish(obj)
    obj['gs_long_hem']=float(end);obj['gs_side_vents']=True

def shalwar(obj):
    """New bifurcated pattern, NOT a scaled Western trouser.

    Gathered waist -> generous seat -> lowered shared crotch saddle -> two
    independent loose thighs -> gradual knee/calf taper -> narrow ankle cuffs.
    Crotch vertices are shared, with pelvis-dominant skinning to limit tearing.
    """
    if obj.data.shape_keys:obj.shape_key_clear()
    if obj.vertex_groups.get('mixamorig:Hips') is None:obj.vertex_groups.new(name='mixamorig:Hips')
    arm=bpy.data.objects['Armature'];bone=lambda n:arm.data.bones['mixamorig:'+n].head_local
    hip=bone('Hips').y;knee=bone('LeftLeg').y;ankle=bone('LeftFoot').y+4
    verts=[];faces=[];weights=[]
    def vertex(co,w):verts.append(co);weights.append(w);return len(verts)-1
    def bridge(a,b):
        for i in range(len(a)):j=(i+1)%len(a);faces.append((a[i],a[j],b[j],b[i]))
    # Circumference starts at rear centre; theta -90..270deg.
    ring=None
    for y,rx,rz in [(hip+4,34,25),(hip,35,26),(hip-9,42,29),(hip-22,47,31),(hip-38,48,29),(hip-51,46,25)]:
        nxt=[]
        for i in range(64):
            theta=-math.pi/2+2*math.pi*i/64
            gather=1+.025*math.cos(12*theta)*smooth((hip+4-y)/20)
            x=rx*math.cos(theta)*gather;z=rz*math.sin(theta)*gather-1
            leg=.38*smooth((hip-y)/51);left=smooth((x+14)/28)
            nxt.append(vertex((x,y,z),{'Hips':1-leg,'LeftUpLeg':leg*left,'RightUpLeg':leg*(1-left)}))
        if ring:bridge(ring,nxt)
        ring=nxt
    # Front-to-back internal saddle shared by BOTH leg tubes.
    saddle=[ring[32]]
    for i in range(1,8):
        t=i/8;saddle.append(vertex((0,hip-51-10*math.sin(math.pi*t),24-50*t),{'Hips':.8,'LeftUpLeg':.1,'RightUpLeg':.1}))
    saddle.append(ring[0])
    for side,sign,arc in [('Left',1,ring[:33]),('Right',-1,[ring[0]]+list(reversed(ring[32:])) )]:
        previous=arc+saddle[1:-1]
        assert len(previous)==40
        # Outer semicircle has 32 sectors, inner saddle eight.
        angles=[-math.pi/2+math.pi*i/32 for i in range(33)]+[math.pi/2+math.pi*i/8 for i in range(1,8)]
        centre=bone(side+'UpLeg').x
        levels=[(hip-65,24,25),(hip-78,23,24),(knee+12,21,22),(knee,19,20),
                (knee-14,17,18),(knee-31,14,15),(ankle+24,11,12),(ankle+8,8,9),(ankle+3,7.5,8),(ankle,7.5,8)]
        for y,rx,rz in levels:
            nxt=[]
            for theta in angles:
                x=centre+sign*rx*math.cos(theta);z=-2+rz*math.sin(theta)
                lower=smooth((knee+16-y)/32);pelvis=.10*(1-smooth((hip-65-y)/20))
                nxt.append(vertex((x,y,z),{'Hips':pelvis,side+'UpLeg':(1-pelvis)*(1-lower),side+'Leg':(1-pelvis)*lower}))
            bridge(previous,nxt);previous=nxt
    # Blender 5 stores group definitions with the mesh; preserve the copied
    # template's group schema while replacing geometry, not its datablock.
    group_names=[g.name for g in obj.vertex_groups]
    mesh=obj.data;mesh.clear_geometry();mesh.name='GS Shalwar Pattern v1'
    mesh.from_pydata(verts,[],faces);mesh.update()
    for name in group_names:
        if obj.vertex_groups.get(name) is None:obj.vertex_groups.new(name=name)
    for i,w in enumerate(weights):
        for name,value in w.items():
            if value>1e-6:
                group=obj.vertex_groups.get('mixamorig:'+name)
                if group is None:raise ValueError('Shalwar rig missing '+name)
                group.add([i],value,'REPLACE')
    # Preview UVs; real front/back photo importer replaces these on build.
    uv=mesh.uv_layers.new(name='GS Photo UV')
    for loop in mesh.loops:
        x,y,z=verts[loop.vertex_index];uv.data[loop.index].uv=((x+50)/100,(y-ankle)/(hip+4-ankle))
    finish(obj);obj['gs_crotch']='shared pelvis-weighted saddle';obj['gs_loose_pattern']=True

def extend_kurta_fit(weights,coords,rest,ids):
    """Smooth hip-to-knee landmark influence only below the pelvic line."""
    hip=(rest[23][1]+rest[24][1])*.5;knee=(rest[25][1]+rest[26][1])*.5
    t=np.clip((hip-coords[:,1])/max(hip-knee,1),0,1)*.50
    extra=[weights[:,2]*t,weights[:,3]*t]
    weights[:,2]*=1-t;weights[:,3]*=1-t
    return np.c_[weights,*extra],ids+[25,26]

def prepare_groups(active,source,category):
    """Append leg groups for long uppers; preserve all existing group indices."""
    import waist_layer
    current=[g.name for g in active.vertex_groups if g.name!=waist_layer.GROUP]
    expected=[g.name for g in source.vertex_groups]
    if current==expected:return
    extras=EXTRA_TOP_GROUPS if active.name=='Tops' else ('mixamorig:Hips',)
    base=lambda names:[n for n in names if n not in extras]
    if active.name not in ('Tops','Bottoms') or base(current)!=base(expected):
        raise ValueError('Incompatible garment rig group order')
    # Blender 5 copies the source group schema with active.data below. Do not
    # mutate the previous live mesh before replacement (rollback must be exact).

def remap_groups(active,source):
    mapping={g.index:active.vertex_groups[g.name].index for g in source.vertex_groups}
    if all(a==b for a,b in mapping.items()):return
    bm=bmesh.new();bm.from_mesh(active.data);layer=bm.verts.layers.deform.verify()
    for v in bm.verts:
        values=dict(v[layer]);v[layer].clear()
        for index,w in values.items():v[layer][mapping[index]]=w
    bm.to_mesh(active.data);bm.free()
