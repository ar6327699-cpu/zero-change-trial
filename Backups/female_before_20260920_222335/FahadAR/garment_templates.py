"""Local rigged garment templates derived only from the user's AR-copy meshes."""
import bpy,bmesh,math
import eastern_foundation as eastern
import garment_visibility as visibility
from mathutils import Vector

CATEGORIES={'HALF_SHIRT':('Half-sleeve shirt','Tops'),
            'FULL_SHIRT':('Full-sleeve shirt','Tops'),
            'SHORTS':('Shorts','Bottoms'),
            'PANTS':('Full pants / tapered','Bottoms'),
            'TROUSERS':('Full trousers / relaxed','Bottoms')}
PREFIX='GS Template '
CATEGORIES.update(eastern.CATEGORIES)  # Append only: preserve saved Western enum indices.

def storage():
    col=bpy.data.collections.get('GS Protected Templates')
    if col is None:
        col=bpy.data.collections.new('GS Protected Templates')
    owner=visibility.studio_scene()
    if col.name not in owner.collection.children:owner.collection.children.link(col)
    col.hide_render=True;col.hide_viewport=True
    return col

def clone(source,name,col):
    obj=source.copy();obj.data=source.data.copy();obj.name=name
    try:visibility.store(obj,col)
    except Exception:
        bpy.data.objects.remove(obj,do_unlink=True)
        raise
    return obj

def extend_mesh(obj,category):
    """Cut closed cuffs/hem and extend native deform-weighted rings.

    Existing upper torso/hip vertices, materials and bone-group order survive.
    This is template modelling, not reconstruction of fabric from photographs.
    """
    if obj.data.shape_keys:obj.shape_key_clear()
    bm=bmesh.new();bm.from_mesh(obj.data)
    deform=bm.verts.layers.deform.verify()
    groups={g.name:g.index for g in obj.vertex_groups}
    arm=bpy.data.objects['Armature']
    bone=lambda n:arm.data.bones['mixamorig:'+n].head_local.copy()
    def assign(v,names,values):
        d=v[deform];d.clear()
        for name,val in zip(names,values):
            if val>1e-6:d[groups['mixamorig:'+name]]=float(val)
    def extend_cut(axis,cut,normal,end,side,taper):
        co=Vector((0,0,0));co[axis]=cut
        bmesh.ops.bisect_plane(bm,geom=list(bm.verts)+list(bm.edges)+list(bm.faces),
            dist=.0001,plane_co=co,plane_no=normal,clear_outer=True,clear_inner=False)
        ring=[e for e in bm.edges if e.is_boundary and all(abs(v.co[axis]-cut)<.002 for v in e.verts)]
        if not ring:raise RuntimeError('Template cut did not produce a cuff boundary')
        origin={v:v.co.copy() for e in ring for v in e.verts}
        if axis==0:
            centre=bone(side+'Arm');joint=bone(side+'ForeArm')[0]
        else:
            centre=bone(side+'Leg');joint=centre[1]
        # 10 rings give a continuous elbow/knee transition at low vertex cost.
        for step in range(1,11):
            fraction=step/10
            result=bmesh.ops.extrude_edge_only(bm,edges=ring)
            new=[v for v in result['geom'] if isinstance(v,bmesh.types.BMVert)]
            newset=set(new)
            for v in new:
                old=v.co.copy();previous=(step-1)/10
                old_taper=1+(taper-1)*previous;new_taper=1+(taper-1)*fraction
                v.co[axis]=cut+(end-cut)*fraction
                for k in range(3):
                    if k!=axis:v.co[k]=centre[k]+(old[k]-centre[k])*new_taper/old_taper
                q=v.co[axis]
                if axis==0:
                    blend=max(0,min(1,(abs(q)-abs(joint)+10)/20))
                    assign(v,[side+'Arm',side+'ForeArm'],[1-blend,blend])
                else:
                    blend=max(0,min(1,(joint+12-q)/24))
                    assign(v,[side+'UpLeg',side+'Leg'],[1-blend,blend])
            ring=[e for e in result['geom'] if isinstance(e,bmesh.types.BMEdge) and all(v in newset for v in e.verts)]
        return len(origin)
    if category=='FULL_SHIRT':
        for side,sign in [('Left',1),('Right',-1)]:
            cut=sign*60
            end=bone(side+'Hand')[0]-sign*3
            extend_cut(0,cut,Vector((sign,0,0)),end,side,.60)
    else:
        # Cut both legs at one level, then extend independently; do not weld
        # across the crotch. Existing pocket openings are not extrusion targets.
        cut=130
        bmesh.ops.bisect_plane(bm,geom=list(bm.verts)+list(bm.edges)+list(bm.faces),
            dist=.0001,plane_co=(0,cut,0),plane_no=(0,-1,0),clear_outer=True)
        for side,sign in [('Left',1),('Right',-1)]:
            ring=[e for e in bm.edges if e.is_boundary and all(abs(v.co.y-cut)<.002 and v.co.x*sign>0 for v in e.verts)]
            if not ring:raise RuntimeError('No trouser hem boundary')
            centre=bone(side+'Leg');end=bone(side+'Foot').y+4
            taper=.66 if category=='PANTS' else .87
            for step in range(1,13):
                t=step/12;previous=(step-1)/12
                result=bmesh.ops.extrude_edge_only(bm,edges=ring)
                new=[v for v in result['geom'] if isinstance(v,bmesh.types.BMVert)];newset=set(new)
                for v in new:
                    v.co.y=cut+(end-cut)*t
                    for k in [0,2]:v.co[k]=centre[k]+(v.co[k]-centre[k])*(1+(taper-1)*t)/(1+(taper-1)*previous)
                    blend=max(0,min(1,(centre.y+12-v.co.y)/24))
                    assign(v,[side+'UpLeg',side+'Leg'],[1-blend,blend])
                ring=[e for e in result['geom'] if isinstance(e,bmesh.types.BMEdge) and all(v in newset for v in e.verts)]
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(obj.data);bm.free();obj.data.update()
    for p in obj.data.polygons:p.use_smooth=True

def ensure_templates():
    col=storage()
    for slot in ['Tops','Bottoms']:
        if bpy.data.objects.get('GS Original '+slot) is None:
            clone(bpy.data.objects[slot],'GS Original '+slot,col)
    for category,(_,slot) in CATEGORIES.items():
        if bpy.data.objects.get(PREFIX+category):continue
        obj=clone(bpy.data.objects['GS Original '+slot],PREFIX+category,col)
        if category in ['FULL_SHIRT','PANTS','TROUSERS']:extend_mesh(obj,category)
        if category=='KURTA':eastern.kurta(obj,extend_mesh)
        if category=='SHALWAR':eastern.shalwar(obj)
        if category=='PAJAMA':
            import eastern_master
            eastern_master.pajama(obj)
        obj['gs_category']=category
    return col
