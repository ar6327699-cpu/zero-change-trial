"""Clean straight-cut Pakistani suit; a new mesh, not the old fitted pattern.

All three garments share the existing Mixamo rig. The separate scarf uses
deterministic folds and torso weights, with no unstable cloth simulation.
"""
import bpy,bmesh,math,json,hashlib
import numpy as np
from pathlib import Path
from mathutils import Matrix

ROOT=Path(__file__).resolve().parent.parent
VERSION=1

class Mesh:
    def __init__(self):self.v=[];self.f=[];self.weights=[];self.regions=[];self.uv=[]
    def vertex(self,co,w):self.v.append(co);self.weights.append(w);return len(self.v)-1
    def face(self,ids,region=0,uv=None):self.f.append(tuple(ids));self.regions.append(region);self.uv.append(uv)
    def bridge(self,a,b,region=0,skip=()):
        for i in range(len(a)):
            if i not in skip:self.face((a[i],a[(i+1)%len(a)],b[(i+1)%len(a)],b[i]),region)

def smooth(t):t=float(np.clip(t,0,1));return t*t*(3-2*t)

def kameez():
    p=Mesh();rings=[];n=64
    levels=[(80,39,21),(82,39,21),(98,38.6,21),(120,38,21),(145,37.2,21),
            (169,36.5,21),(184,36,21),(205,35.5,21),(228,35,21),(250,35,21.5),
            (269,35.5,22),(284,36.5,22),(295,36,20),(306,32,17),(316,12,10)]
    for row,(y,rx,rz) in enumerate(levels):
        ring=[]
        for i in range(n):
            a=math.tau*i/n;x=rx*math.cos(a);front=math.sin(a)
            fold=(.30*math.sin(6*a)+.18*math.sin(10*a))*smooth((260-y)/140)
            z=-5+(rz+fold)*front;yy=y
            if row==len(levels)-1:yy-=9*max(0,front)**1.4+2*max(0,-front)+3*math.exp(-(x/2.5)**2)*max(0,front)
            if row<2:yy-=.55*(1+math.cos(a*32))
            # Broad torso weights, no hip-width shaping and no leg-driven hem.
            t=smooth((y-205)/80);w={'Hips':1-t,'Spine2':t}
            if y>=284:
                arm=.55*smooth((abs(x)-23)/15)
                w={'Spine2':1-arm,('LeftArm' if x>0 else 'RightArm'):arm}
            ring.append(p.vertex((x,yy,z),w))
        rings.append(ring)
    hole=set(range(60,64))|set(range(4))|set(range(28,36))
    for k in range(len(levels)-1):
        skip=set(hole) if levels[k][0] in (284,295) else set()
        if levels[k][0]<184:skip|={0,31,32,63}
        p.bridge(rings[k],rings[k+1],skip=skip)
    counts={}
    for f in p.f:
        for a,b in zip(f,f[1:]+f[:1]):counts[tuple(sorted((a,b)))]=counts.get(tuple(sorted((a,b))),0)+1
    for sign,side,region in ((1,'Left',1),(-1,'Right',2)):
        edges=[e for e,c in counts.items() if c==1 and all(283<p.v[i][1]<307 and sign*p.v[i][0]>23 for i in e)]
        adj={}
        for a,b in edges:adj.setdefault(a,[]).append(b);adj.setdefault(b,[]).append(a)
        assert adj and all(len(v)==2 for v in adj.values()),'Open armhole boundary'
        ring=[min(adj)];previous=None;now=ring[0]
        while True:
            nxt=next(v for v in adj[now] if v!=previous)
            if nxt==ring[0]:break
            ring.append(nxt);previous,now=now,nxt
        angles=np.arctan2(np.array([p.v[i][2]+5 for i in ring]),np.array([p.v[i][1]-295 for i in ring]))
        prev=ring
        for k in range(1,23):
            t=k/22;x=sign*(36+107*t);cy=295+5*smooth(t*4)
            r=14.5-2*math.sin(math.pi*t)+1.4*smooth((t-.75)/.25)
            elbow=abs(bpy.data.objects['Armature'].data.bones['mixamorig:'+side+'ForeArm'].head_local.x)
            lower=smooth((abs(x)-elbow+25)/50)
            ring=[p.vertex((x,cy+r*math.cos(a),-10+r*math.sin(a)),{side+'Arm':1-lower,side+'ForeArm':lower}) for a in angles]
            p.bridge(prev,ring,region);prev=ring
    return p

def trousers():
    p=Mesh();n=64;prev=None
    for y,rx,rz in [(212,30,18),(209,30.5,19),(195,32,21),(181,32,21),(169,31.5,20)]:
        ring=[]
        for i in range(n):
            a=-math.pi/2+math.tau*i/n;x=rx*math.cos(a);side=smooth((x+22)/44);leg=.2*smooth((211-y)/42)
            ring.append(p.vertex((x,y,-5+rz*math.sin(a)),{'Hips':1-leg,'LeftUpLeg':leg*side,'RightUpLeg':leg*(1-side)}))
        if prev:p.bridge(prev,ring)
        prev=ring
    seat=[ring[32]]
    for i in range(1,9):
        t=i/9;seat.append(p.vertex((0,169-7*math.sin(math.pi*t),15-40*t),{'Hips':.7,'LeftUpLeg':.15,'RightUpLeg':.15}))
    seat.append(ring[0])
    angles=[-math.pi/2+math.pi*i/32 for i in range(33)]+[math.pi/2+math.pi*i/9 for i in range(1,9)]
    for sign,side,arc in [(1,'Left',ring[:33]),(-1,'Right',[ring[0]]+list(reversed(ring[32:])) )]:
        prev=arc+seat[1:-1]
        for y in [155,141,127,117,110,103,91,78,65,52,40,28,26]:
            r=float(np.interp(y,[26,110,155],[10.5,12,15]))
            knee=smooth((128-y)/36);hip=.12*smooth((y-136)/20)
            row=[p.vertex((sign*(18.5+r*math.cos(a)),y,-5+r*1.03*math.sin(a)),{'Hips':hip,side+'UpLeg':(1-hip)*(1-knee),side+'Leg':(1-hip)*knee}) for a in angles]
            p.bridge(prev,row);prev=row
    return p

def dupatta():
    p=Mesh();rows=[];nu=24;nv=112
    # One continuous U-drape: front fall, shoulder bridge, back fall.
    for j in range(nv+1):
        t=j/nv
        if t<.465:
            q=t/.465;y=27+279*q;z=26-11*q;width=39-24*q
        elif t>.535:
            q=(t-.535)/.465;y=306-271*q;z=-24-11*q;width=15+22*q
        else:
            q=(t-.465)/.07;y=306+4*math.sin(math.pi*q);z=15-39*q;width=15
        row=[]
        for i in range(nu+1):
            u=i/nu;fold=2.2*math.sin(u*math.tau*3+.6*math.sin(t*8))+.55*math.sin(u*math.tau*7)
            x=28+(u-.5)*width+2.5*math.sin(t*7)
            yy=y-.35*(x-26)*smooth((y-270)/30)+(.55*math.cos(u*math.tau*8) if j in (0,nv) else 0)
            blend=smooth((y-210)/80)
            sweep=-(u-.5)*7 if t<.465 else (u-.5)*7 if t>.535 else 0
            row.append(p.vertex((x,yy,z+fold+sweep),{'Hips':1-blend,'Spine2':blend}))
        rows.append(row)
    for j in range(nv):
        for i in range(nu):
            p.face((rows[j][i],rows[j][i+1],rows[j+1][i+1],rows[j+1][i]),3,[(i/nu,1-j/nv),((i+1)/nu,1-j/nv),((i+1)/nu,1-(j+1)/nv),(i/nu,1-(j+1)/nv)])
    # Anchor the bridge to the actual new garment surface once at authoring.
    # This is static placement, not a live nearest-surface solver.
    from mathutils.bvhtree import BVHTree
    from mathutils import Vector
    shirt=kameez();tree=BVHTree.FromPolygons(shirt.v,shirt.f,all_triangles=False)
    for index,co in enumerate(p.v):
        x,y,z=co
        if y>298:
            hit,normal,face,distance=tree.ray_cast(Vector((x,360,z)),Vector((0,-1,0)))
            if hit is not None:p.v[index]=(x,hit.y+.9,z)
    return p

def fabric(name,image,transparent=False):
    old=bpy.data.materials.get(name)
    if old:return old
    m=bpy.data.materials.new(name);m.use_nodes=True;m.use_backface_culling=False
    bs=m.node_tree.nodes.get('Principled BSDF');bs.inputs['Roughness'].default_value=.86
    bs.inputs['Metallic'].default_value=0;bs.inputs['Coat Weight'].default_value=0;bs.inputs['Specular IOR Level'].default_value=.18
    tex=m.node_tree.nodes.new('ShaderNodeTexImage');im=bpy.data.images.load(str(ROOT/'References/MintReference'/image),check_existing=True);im.pack();tex.image=im
    m.node_tree.links.new(tex.outputs['Color'],bs.inputs['Base Color'])
    if transparent:
        bs.inputs['Alpha'].default_value=.84;m.surface_render_method='BLENDED'
        if hasattr(m,'use_transparency_overlap'):m.use_transparency_overlap=False
    return m

def create(name,p,category,collection):
    old=bpy.data.objects.get(name)
    if old:bpy.data.objects.remove(old,do_unlink=True)
    mesh=bpy.data.meshes.new(name+' clean topology');mesh.from_pydata(p.v,[],p.f);mesh.update()
    obj=bpy.data.objects.new(name,mesh);collection.objects.link(obj);obj.parent=bpy.data.objects['Armature']
    obj.matrix_parent_inverse=Matrix.Identity(4);obj.matrix_basis=Matrix.Identity(4)
    obj.modifiers.new('Shared armature','ARMATURE').object=obj.parent
    for bone in obj.parent.data.bones:obj.vertex_groups.new(name=bone.name)
    for i,w in enumerate(p.weights):
        for bone,value in w.items():
            if value>1e-8:obj.vertex_groups['mixamorig:'+bone].add([i],value/sum(w.values()),'REPLACE')
    region=mesh.attributes.new('GS_pattern_region','INT','FACE');region.data.foreach_set('value',p.regions)
    uv=mesh.uv_layers.new(name='GS Photo UV')
    mats=[fabric('Mint floral front','front.png'),fabric('Mint motifs back','back.png'),fabric('Mint embroidered sleeves','sleeve.png')]
    if category=='FEMALE_TROUSER':mats=[fabric('Mint straight trouser','trouser.png')]
    if category=='FEMALE_DUPATTA':mats=[fabric('Pink chiffon dupatta','dupatta.png',True)]
    for m in mats:mesh.materials.append(m)
    for poly in mesh.polygons:
        coords=np.array([p.v[i] for i in poly.vertices]);reg=p.regions[poly.index]
        back=coords[:,2].mean()<-5
        poly.material_index=2 if category=='FEMALE_KAMEEZ' and reg else int(back) if category=='FEMALE_KAMEEZ' else 0
        for k,li in enumerate(poly.loop_indices):
            x,y,z=p.v[mesh.loops[li].vertex_index]
            if category=='FEMALE_DUPATTA':val=p.uv[poly.index][k]
            elif category=='FEMALE_KAMEEZ' and reg:
                val=(.5+((y-300)/32)*.95,1-np.clip((abs(x)-36)/107,0,1))
            elif category=='FEMALE_KAMEEZ':val=(.5+(-x if back else x)/80,np.clip((y-80)/236,0,1))
            else:val=(.5+x/65,np.clip((y-26)/186,0,1))
            uv.data[li].uv=val
    bm=bmesh.new();bm.from_mesh(mesh);bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(mesh);bm.free()
    for poly in mesh.polygons:poly.use_smooth=True
    if category!='FEMALE_DUPATTA':
        smoothmod=obj.modifiers.new('Tailored surface','SUBSURF');smoothmod.levels=1;smoothmod.render_levels=1
    obj['gs_category']=category;obj['gs_profile']='FEMALE';obj['gs_clean_reference_version']=VERSION
    obj['gs_label']={'FEMALE_KAMEEZ':'Mint floral straight Kameez','FEMALE_TROUSER':'Mint straight embroidered Trouser','FEMALE_DUPATTA':'Pink separate shoulder Dupatta'}[category]
    obj['gs_photo_method']='New clean reference pattern; fixed front/back/sleeve UVs'
    obj['gs_design_parameters']=json.dumps({'silhouette':'straight loose','hem':80,'reference':'MintReference/reference.png','physics':'none'})
    obj.hide_viewport=True;obj.hide_render=True
    if category!='FEMALE_DUPATTA':
        import female_patterns
        female_patterns.regional_keys(obj)
    return obj

def build_library():
    import garment_studio as gs
    specs=[('FEMALE_KAMEEZ',kameez),('FEMALE_TROUSER',trousers),('FEMALE_DUPATTA',dupatta)]
    rows=gs.manifest();ids=[]
    for cat,fn in specs:
        identifier=hashlib.md5(('mint_clean_v1_'+cat).encode()).hexdigest();ids.append(identifier)
        obj=create('GS Item '+identifier,fn(),cat,gs.templates.storage());obj['gs_id']=identifier
        bpy.data.libraries.write(str(gs.LIBRARY/(identifier+'.blend')),{obj},fake_user=True)
        rows=[r for r in rows if r['id']!=identifier]
        rows.append({'id':identifier,'name':'FEMALE | '+obj['gs_label'],'category':cat,'profile':'FEMALE','method':obj['gs_photo_method'],'notes':['New mesh from mint four-view reference'],'photo_layout':'REFERENCE'})
    (gs.LIBRARY/'library.json').write_text(json.dumps(rows,indent=2))
    outfits=[r for r in gs.outfits() if r['key']!='female_mint_reference']
    outfits.insert(0,{'key':'female_mint_reference','name':'Mint floral + pink Dupatta','profile':'FEMALE','top':ids[0],'bottom':ids[1],'dupatta':ids[2]})
    (gs.LIBRARY/'outfits.json').write_text(json.dumps(outfits,indent=2))
    return ids
