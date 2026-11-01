"""Eastern-only live presentation. Camera pixels and Western defaults are unchanged."""
import bpy
import numpy as np

LIGHT_KEY='gs_eastern_original_energy'
WORLD_KEY='gs_eastern_original_strength'
VIEW_KEY='gs_eastern_original_light_eyes'
HEM='AR Eastern Knee Hem'

def lighting(scene):
    """Undo the over-lit legacy AR rig, not the source photo or camera exposure.

    Stored originals make this idempotent across restart/save and reversible
    when the last Eastern garment is removed. Only the two owned AR lamps and
    owned neutral world are changed; unrelated user lights are never edited.
    """
    enabled=any(bpy.data.objects.get(n) and bpy.data.objects[n].get('gs_category') in
                ('KURTA','SHALWAR','PAJAMA','FEMALE_KAMEEZ','FEMALE_TROUSER','FEMALE_SHALWAR') for n in ('Tops','Bottoms'))
    # Legacy avatar lamps are render-disabled but can still illuminate Material
    # Preview when their per-layer eye has been reopened. Keep the two pipelines
    # consistent without touching user-created lights or other scenes.
    from garment_visibility import hide_in_layer
    for name in ('Key','Fill','Rim'):
        obj=scene.objects.get(name)
        if obj is None or obj.type!='LIGHT':continue
        if enabled and obj.hide_render:
            previous=dict(obj.get(VIEW_KEY,{}))
            for layer in scene.view_layers:
                layer.update()
                if layer.objects.get(obj.name)!=obj:continue
                if layer.name not in previous:previous[layer.name]=obj.hide_get(view_layer=layer)
                hide_in_layer(obj,True,layer)
            obj[VIEW_KEY]=previous
        elif not enabled and VIEW_KEY in obj:
            for name,hidden in obj[VIEW_KEY].items():
                layer=scene.view_layers.get(name)
                if layer:hide_in_layer(obj,bool(hidden),layer)
            del obj[VIEW_KEY]
    for name in ('AR Key','AR Fill'):
        obj=scene.objects.get(name)
        if obj is None or obj.type!='LIGHT':continue
        data=obj.data
        if enabled:
            if LIGHT_KEY not in data:data[LIGHT_KEY]=float(data.energy)
            data.energy=float(data[LIGHT_KEY])*.35
        elif LIGHT_KEY in data:
            data.energy=float(data[LIGHT_KEY]);del data[LIGHT_KEY]
    world=scene.world
    if world and world.name.startswith('AR Neutral World') and world.use_nodes:
        node=world.node_tree.nodes.get('Background')
        if node:
            strength=node.inputs['Strength']
            if enabled:
                if WORLD_KEY not in world:world[WORLD_KEY]=float(strength.default_value)
                strength.default_value=float(world[WORLD_KEY])*.35
            elif WORLD_KEY in world:
                strength.default_value=float(world[WORLD_KEY]);del world[WORLD_KEY]
    return enabled

def knee_modifier(obj,rest,coords):
    """Post-skinning graded panel length, without pulling sleeves/torso down.

    At the hem the height is exactly the observed knee plane; the correction
    fades quadratically to zero at the hip. This is a visual length constraint,
    not cloth physics. It uses evaluated positions, so pose/scale are included.
    """
    if obj.name!='Tops' or obj.get('gs_category')!='KURTA':return None
    hip=float((rest[23][1]+rest[24][1])*.5);hem=float(coords[:,1].min())
    fraction=np.clip((hip-coords[:,1])/max(hip-hem,1),0,1)
    attr=obj.data.attributes.get('GS_hem_fraction') or obj.data.attributes.new('GS_hem_fraction','FLOAT','POINT')
    attr.data.foreach_set('value',fraction.astype(np.float32))
    group=bpy.data.node_groups.get(HEM)
    if group is None:
        group=bpy.data.node_groups.new(HEM,'GeometryNodeTree')
        for name,typ,io in [('Geometry','NodeSocketGeometry','INPUT'),('Geometry','NodeSocketGeometry','OUTPUT'),
                            ('Hip','NodeSocketFloat','INPUT'),('Knee','NodeSocketFloat','INPUT'),
                            ('World row','NodeSocketVector','INPUT'),('World offset','NodeSocketFloat','INPUT'),
                            ('Local up','NodeSocketVector','INPUT'),('Enabled','NodeSocketFloat','INPUT')]:
            group.interface.new_socket(name=name,in_out=io,socket_type=typ)
        n=group.nodes;l=group.links;inp=n.new('NodeGroupInput');out=n.new('NodeGroupOutput')
        pos=n.new('GeometryNodeInputPosition');attrnode=n.new('GeometryNodeInputNamedAttribute');attrnode.data_type='FLOAT';attrnode.inputs['Name'].default_value='GS_hem_fraction'
        def mathnode(op,a,b):
            node=n.new('ShaderNodeMath');node.operation=op
            for i,v in enumerate((a,b)):
                if isinstance(v,(int,float)):node.inputs[i].default_value=v
                else:l.new(v,node.inputs[i])
            return node.outputs[0]
        dot=n.new('ShaderNodeVectorMath');dot.operation='DOT_PRODUCT';l.new(pos.outputs[0],dot.inputs[0]);l.new(inp.outputs['World row'],dot.inputs[1])
        current=mathnode('ADD',dot.outputs['Value'],inp.outputs['World offset'])
        t=attrnode.outputs['Attribute'];target=mathnode('ADD',inp.outputs['Hip'],mathnode('MULTIPLY',t,mathnode('SUBTRACT',inp.outputs['Knee'],inp.outputs['Hip'])))
        delta=mathnode('MULTIPLY',mathnode('SUBTRACT',target,current),mathnode('MULTIPLY',mathnode('MULTIPLY',t,t),inp.outputs['Enabled']))
        scale=n.new('ShaderNodeVectorMath');scale.operation='SCALE';l.new(inp.outputs['Local up'],scale.inputs[0]);l.new(delta,scale.inputs['Scale'])
        setpos=n.new('GeometryNodeSetPosition');l.new(inp.outputs['Geometry'],setpos.inputs['Geometry']);l.new(scale.outputs['Vector'],setpos.inputs['Offset']);l.new(setpos.outputs[0],out.inputs['Geometry'])
    mod=obj.modifiers.get(HEM) or obj.modifiers.new(HEM,'NODES');mod.node_group=group
    obj.modifiers.move(list(obj.modifiers).index(mod),len(obj.modifiers)-1)
    sockets={s.name:s.identifier for s in group.interface.items_tree if s.item_type=='SOCKET' and s.in_out=='INPUT'}
    mod[sockets['Enabled']]=0.
    return mod,sockets

def update_hem(obj,state,matrix,p,projected):
    from mathutils import Vector
    if state is None:return
    mod,sockets=state
    # Preserve the authored knee/below-knee length. Driving the whole skirt to
    # the higher tracked knee amputates the hem during steps and squats.
    mod[sockets['Enabled']]=0.;obj.update_tag();return
    # Use confident visible knees. During loss, use current projected skeleton,
    # never a stale frame or an extrapolated arbitrary percentage of torso size.
    def height(i):
        if np.isfinite(p[i]).all() and p[i,3]>=.55 and .005<p[i,1]<.995:
            return (.5-float(p[i,1]))*2
        return float(projected[i][2])
    hip=(height(23)+height(24))*.5
    knee=max(height(25),height(26))
    knee+=max(0.,hip-knee)*.025  # small above-knee safety margin
    values={'Hip':hip,'Knee':knee,'World row':tuple(matrix[2][:3]),'World offset':float(matrix[2][3]),
            'Local up':tuple(matrix.inverted().to_3x3()@Vector((0,0,1))),'Enabled':1.}
    for name,value in values.items():mod[sockets[name]]=value
    obj.update_tag()
