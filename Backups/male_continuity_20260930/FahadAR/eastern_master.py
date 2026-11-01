"""Versioned, person-independent Eastern masters. No Western mesh edits.

Coordinates use the existing rig's local units; depth is a pattern prior, not
an inferred body measurement. A derived asset keeps master topology/weights.
"""
import math
import bpy
import numpy as np
import eastern_foundation as foundation

VERSION=3
PREFIX='GS Master v3 '

def install_mesh(obj,verts,faces,weights,regions):
    used=sorted({i for face in faces for i in face});remap={old:new for new,old in enumerate(used)}
    verts=[verts[i] for i in used];weights=[weights[i] for i in used]
    faces=[tuple(remap[i] for i in face) for face in faces]
    names=[g.name for g in obj.vertex_groups]
    mesh=obj.data
    if mesh.shape_keys:obj.shape_key_clear()
    mesh.clear_geometry();mesh.from_pydata(verts,[],faces);mesh.update()
    for name in names:
        if obj.vertex_groups.get(name) is None:obj.vertex_groups.new(name=name)
    for i,values in enumerate(weights):
        total=sum(values.values())
        for name,value in values.items():
            if value<=1e-8:continue
            name='mixamorig:'+name
            group=obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
            group.add([i],value/total,'REPLACE')
    foundation.finish(obj)
    attr=mesh.attributes.new('GS_pattern_region','INT','FACE')
    attr.data.foreach_set('value',regions)
    mesh['gs_master_version']=VERSION
    obj['gs_foundation']='male_eastern_master_v3'

def kurta(obj):
    """Sewn volumetric torso, open hem/side vents, connected arms and band neck."""
    verts=[];faces=[];weights=[];regions=[]
    def add(co,w):
        verts.append(co);weights.append(w);return len(verts)-1
    def face(ids,region=0):faces.append(ids);regions.append(region)
    # Restrained lower flare, slight waist suppression and rounded chest;
    # preserve shoulder pivots and sewn armholes rather than widening a shirt.
    levels=[(118,48,31),(121,48,31),(140,47.5,31),(164,47,31),
            (186,47,31),(199,47,31),(216,46,30),(236,43,28),
            (257,44,28),(280,46,27),(295,47,25),(310,42,22),
            (318,12,11),(324,12,11)]
    rings=[];count=64
    for y,rx,rz in levels:
        ring=[]
        for i in range(count):
            theta=2*math.pi*i/count;x=rx*math.cos(theta);z=-5+rz*math.sin(theta)
            # Restrained vertical folds, strongest below the hip, not noisy wrinkles.
            fold=.7*foundation.smooth((230-y)/80)*math.cos(6*theta)
            z+=fold*math.sin(theta)
            if y<236:
                spine=.35*foundation.smooth((y-205)/31)
                thigh=.12*foundation.smooth((199-y)/84)
                side=foundation.smooth((x+18)/36)
                w={'Hips':1-spine-thigh,'Spine':spine,
                   'LeftUpLeg':thigh*side,'RightUpLeg':thigh*(1-side)}
            elif y<280:
                blend=foundation.smooth((y-236)/44)
                w={'Spine':1-blend,'Spine2':blend}
            else:
                arm=.55*foundation.smooth((abs(x)-30)/14)
                w={'Spine2':1-arm,('LeftArm' if x>=0 else 'RightArm'):arm}
            ring.append(add((x,y,z),w))
        rings.append(ring)
    for k in range(len(rings)-1):
        y=levels[k][0]
        for i in range(count):
            j=(i+1)%count
            # Two bounded armhole patches, not separate overlapping tubes.
            hole=(i in list(range(60,64))+list(range(4))+list(range(28,36)))
            if k in (9,10) and hole:continue
            # Narrow lateral openings below hip. Front and back remain panels.
            if y<199 and i in (0,31,32,63):continue
            face((rings[k][i],rings[k][j],rings[k+1][j],rings[k+1][i]))
    # Find each armhole boundary by adjacency; loft its ordered ring to cuff.
    edges={}
    for f in faces:
        for a,b in zip(f,f[1:]+f[:1]):
            edge=tuple(sorted((a,b)));edges[edge]=edges.get(edge,0)+1
    for sign,side,region in [(1,'Left',1),(-1,'Right',2)]:
        boundary=[e for e,n in edges.items() if n==1 and
                  all(279<verts[v][1]<311 and verts[v][0]*sign>29 for v in e)]
        adjacent={}
        for a,b in boundary:adjacent.setdefault(a,[]).append(b);adjacent.setdefault(b,[]).append(a)
        if not adjacent or any(len(x)!=2 for x in adjacent.values()):
            raise ValueError('Eastern master armhole is not a closed boundary')
        start=min(adjacent);root=[start];previous=None;current=start
        while True:
            nxt=next(v for v in adjacent[current] if v!=previous)
            if nxt==start:break
            root.append(nxt);previous,current=current,nxt
        assert len(root)==len(adjacent)
        coords=np.array([verts[i] for i in root]);angle=np.arctan2(coords[:,2]+5,coords[:,1]-295)
        last=root
        for step in range(1,13):
            t=step/12;x=sign*(48+95*t);y=295+5*min(1,t*3)
            radius_y=16*(1-t)+8*t;radius_z=18*(1-t)+8*t
            lower=foundation.smooth((abs(x)-84)/20)
            ring=[]
            for a in angle:
                ring.append(add((x,y+radius_y*math.cos(a),-10+radius_z*math.sin(a)),
                                {side+'Arm':1-lower,side+'ForeArm':lower}))
            for i in range(len(last)):
                j=(i+1)%len(last);face((last[i],last[j],ring[j],ring[i]),region)
            last=ring
    install_mesh(obj,verts,faces,weights,regions)
    obj['gs_side_vents']=True;obj['gs_long_hem']=118.
    obj['gs_photo_requirement']='Full front/back product pair, plain contrasting background'

def shalwar(obj):
    # The foundation already uses an independent bifurcated pattern and shared
    # crotch saddle. Preserve that topology; add restrained gathered volume.
    foundation.shalwar(obj)
    c=np.array([v.co[:] for v in obj.data.vertices])
    for i,(x,y,z) in enumerate(c.copy()):
        upper=foundation.smooth((y-62)/78)*(1-foundation.smooth((y-191)/22))
        c[i,0]=x*(1+.10*upper)
        c[i,2]=-2+(z+2)*(1+.08*upper)
        c[i,1]=np.interp(y,[28,110,122,131,144,158,191,213],
                             [28,98,108,118,128,145,191,213])
    obj.data.vertices.foreach_set('co',c.ravel());obj.data.update()
    obj.data['gs_master_version']=VERSION
    obj['gs_foundation']='male_eastern_master_v3'

def pajama(obj):
    """Distinct straight Eastern lower: higher rise, less seat ease, open cuffs.

    Reuses the sewn lower-family topology, not a Western pants mesh and not
    Shalwar's dropped-crotch/balloon dimensions.
    """
    foundation.shalwar(obj)
    c=np.array([v.co[:] for v in obj.data.vertices])
    for i,(x,y,z) in enumerate(c.copy()):
        if y>=158:
            factor=float(np.interp(y,[158,187,213],[.82,.82,1.]))
            c[i,0]=x*factor;c[i,2]=-2+(z+2)*.84
        else:
            centre=18.5 if x>=0 else -18.5
            factor=float(np.interp(y,[28,110,158],[1.25,.80,.76]))
            c[i,0]=centre+(x-centre)*factor;c[i,2]=-2+(z+2)*factor
        c[i,1]=np.interp(y,[28,110,122,131,144,148,158,171,187,200,209,213],
                            [28,110,126,138,167,170,178,185,192,201,209,213])
    obj.data.vertices.foreach_set('co',c.ravel());obj.data.update()
    obj.data['gs_master_version']=VERSION
    obj['gs_foundation']='male_eastern_master_v3'
    obj['gs_lower_cut']='straight pajama: higher crotch, restrained seat, open ankle'

def ensure(category,storage,clone):
    name=PREFIX+category
    obj=bpy.data.objects.get(name)
    if obj is not None:return obj
    slot='Tops' if category=='KURTA' else 'Bottoms'
    obj=clone(bpy.data.objects['GS Original '+slot],name,storage)
    try:
        {'KURTA':kurta,'SHALWAR':shalwar,'PAJAMA':pajama}[category](obj)
        obj['gs_category']=category;obj['gs_master']=True
        return obj
    except Exception:
        bpy.data.objects.remove(obj,do_unlink=True)
        raise
