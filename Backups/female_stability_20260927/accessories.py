"""One separate shoulder accessory, using the shared rig and profile lifecycle."""
import bpy

def attachment_weights(actor,top):
    """Use the upper's rest-surface ownership for a separate shoulder drape.

    Transfer post-skin fit and skin weights from the same upper surface. Free
    hanging ends follow its hem ownership. No changing
    nearest-point queries in live frames, which would make attachment flicker.
    """
    import numpy as np
    from mathutils.kdtree import KDTree
    data=top.data.shape_keys.key_blocks[0].data if top.data.shape_keys else top.data.vertices
    tree=KDTree(len(data))
    for i,v in enumerate(data):tree.insert(v.co,i)
    tree.balance()
    ids=sorted(int(a.name[7:]) for a in top.data.attributes if a.name.startswith('AR_fit_'))
    source=np.array([[top.data.attributes['AR_fit_'+str(i)].data[v.index].value for i in ids] for v in top.data.vertices])
    result=[]
    attachment_key=str(top.get('gs_id',''))+':'+top.data.name
    bind=actor.get('gs_shoulder_attachment_v2')!=attachment_key
    for v in actor.data.vertices:
        nearest=tree.find_n(v.co,4)
        fractions=1/np.maximum([item[2] for item in nearest],.05)**2
        fractions/=fractions.sum()
        result.append(sum(source[item[1]]*fraction for item,fraction in zip(nearest,fractions)))
        # The shoulder AND torso follow the kameez. Keeping the old hips/spine
        # mix below the bridge would separate the drape during a torso bend.
        if bind:
            weights={actor.vertex_groups[g.group].name:0. for g in v.groups}
            for item,fraction in zip(nearest,fractions):
                for g in top.data.vertices[item[1]].groups:
                    name=top.vertex_groups[g.group].name
                    weights[name]=weights.get(name,0)+fraction*g.weight
            for name,value in weights.items():
                group=actor.vertex_groups.get(name) or actor.vertex_groups.new(name=name)
                group.add([v.index],value,'REPLACE')
    actor['gs_shoulder_attachment_v2']=attachment_key
    weights=np.asarray(result);weights/=np.maximum(weights.sum(1,keepdims=True),1e-8)
    return weights,ids

def surface(actor):
    """Apply the reference garment's finish only to its active mesh."""
    if actor.name=='Dupatta':
        import cosmetic_finish
        cosmetic_finish.match_dupatta(actor,bpy.data.objects.get('Tops'))
    mod=actor.modifiers.get('Tailored surface')
    if actor.get('gs_clean_reference_version'):
        if mod is None:mod=actor.modifiers.new('Tailored surface','SUBSURF')
        mod.levels=1;mod.render_levels=1
        mod.show_viewport=True;mod.show_render=True
        # Deform and fit the coarse pattern first, then smooth the final surface.
        actor.modifiers.move(list(actor.modifiers).index(mod),len(actor.modifiers)-1)
    elif mod is not None:actor.modifiers.remove(mod)

def activate(identifier=None):
    import garment_studio as gs,body_profiles as bp
    if bp.current()!='FEMALE':sync();return
    if identifier is None:
        identifier=next((r['id'] for r in gs.manifest() if r['category']=='FEMALE_DUPATTA'),None)
    if identifier is None:return
    source,row=gs.item(identifier)
    if row['category']!='FEMALE_DUPATTA':raise ValueError('Accessory must be a separate Dupatta')
    actor=bpy.data.objects.get('Dupatta')
    if actor is None:
        actor=source.copy();actor.data=source.data.copy();actor.name='Dupatta';gs.scene().collection.objects.link(actor)
    else:
        # Mesh group indices belong to the source schema, not the prior item.
        actor.vertex_groups.clear()
        actor.data=source.data.copy()
        for group in source.vertex_groups:actor.vertex_groups.new(name=group.name)
    if 'gs_shoulder_attachment_v2' in actor:del actor['gs_shoulder_attachment_v2']
    actor['gs_category']='FEMALE_DUPATTA';actor['gs_profile']='FEMALE';actor['gs_id']=identifier
    actor['gs_accessory_enabled']=True;actor.parent=bpy.data.objects['Armature']
    for mod in actor.modifiers:
        if mod.type=='ARMATURE':mod.object=actor.parent;mod.use_deform_preserve_volume=True
    fit_drape(actor,bpy.data.objects.get('Tops'))
    actor.hide_viewport=False;actor.hide_render=True;actor.hide_set(True)
    gs.scene()['GS_female_dupatta']=identifier

def fit_drape(actor,top):
    """Place the separate drape outside the selected upper's rest surface."""
    if not top:return
    import numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    data=top.data.shape_keys.key_blocks[0].data if top.data.shape_keys else top.data.vertices
    tree=BVHTree.FromPolygons([v.co[:] for v in data],[p.vertices[:] for p in top.data.polygons],all_triangles=False)
    # Preserve the authored folds. Clamping each valley individually leaves
    # long flat triangles which can cross the upper between sampled vertices.
    # Translate each complete weave row by the largest required clearance.
    uv=actor.data.uv_layers.active
    rows={}
    for poly in actor.data.polygons:
        for loop in poly.loop_indices:
            vertex=actor.data.loops[loop].vertex_index
            row=round(float(uv.data[loop].uv.y),5)
            rows.setdefault(row,set()).add(vertex)
    for row,ids in rows.items():
        points=np.array([actor.data.vertices[i].co[:] for i in sorted(ids)])
        mean=points.mean(0);bridge=.455<1-row<.545
        if bridge:
            # Some original bridge rays missed the shoulder and hit the hem.
            # Absolute shoulder placement repairs that isolated bad vertex;
            # a row translation would stretch all its neighbours upwards.
            for i in ids:
                v=actor.data.vertices[i];x,y,z=v.co
                hit,*_=tree.ray_cast(Vector((x,370,z)),Vector((0,-1,0)))
                if hit is not None and hit.y>275:v.co.y=hit.y+3.
                else:v.co.y=306-.35*(x-26)
            continue
        axis=2;sign=1 if 1-row<.5 else -1
        shift=0.
        for x,y,z in points:
            origin=Vector((x,y,sign*150))
            direction=Vector((0,0,-sign))
            hit,*_=tree.ray_cast(origin,direction)
            if hit is not None:shift=max(shift,sign*(hit[axis]-z)+4.)
        for i in ids:actor.data.vertices[i].co[axis]+=sign*shift
    actor.data.update()

    import cosmetic_finish
    cosmetic_finish.match_dupatta(actor,top)

def sync():
    import body_profiles as bp
    actor=bpy.data.objects.get('Dupatta')
    if actor:
        actor.hide_render=True;actor.hide_viewport=bp.current()!='FEMALE'
        if bpy.context.view_layer.objects.get(actor.name):actor.hide_set(True)
