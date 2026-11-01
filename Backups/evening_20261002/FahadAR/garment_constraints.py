"""Camera-depth layering and a live lower-garment hem limit.

Native geometry nodes evaluate after skinning and landmark corrections. No
live nearest-surface switching, no mesh/UV rewrite, no cloth simulation.
"""
import bpy
from mathutils import Vector

def make_group(name,extra):
    group=bpy.data.node_groups.get(name)
    if group:return group
    group=bpy.data.node_groups.new(name,'GeometryNodeTree')
    for label,typ,io in [('Geometry','NodeSocketGeometry','INPUT'),('Geometry','NodeSocketGeometry','OUTPUT')]+extra:
        group.interface.new_socket(name=label,socket_type=typ,in_out=io)
    return group

def helpers(group):
    nodes=group.nodes;links=group.links
    def node(kind):return nodes.new(kind)
    def connect(value,socket):
        if hasattr(value,'node'):links.new(value,socket)
        else:socket.default_value=value
    def math(op,a,b):
        n=node('ShaderNodeMath');n.operation=op
        connect(a,n.inputs[0]);connect(b,n.inputs[1]);return n.outputs[0]
    def vector(op,a,b):
        n=node('ShaderNodeVectorMath');n.operation=op
        connect(a,n.inputs[0]);connect(b,n.inputs['Scale'] if op=='SCALE' else n.inputs[1])
        return n.outputs['Value' if op=='DOT_PRODUCT' else 'Vector']
    return node,connect,math,vector

def modifier(obj,group):
    mod=obj.modifiers.get(group.name) or obj.modifiers.new(group.name,'NODES');mod.node_group=group
    obj.modifiers.move(list(obj.modifiers).index(mod),len(obj.modifiers)-1)
    sockets={s.name:s.identifier for s in group.interface.items_tree if s.item_type=='SOCKET' and s.in_out=='INPUT'}
    return obj,mod,sockets

def ensure():
    top=bpy.data.objects.get('Tops');bottom=bpy.data.objects.get('Bottoms');scarf=bpy.data.objects.get('Dupatta');states=[]
    enabled=top and bottom and top.get('gs_category') in ('KURTA','FEMALE_KAMEEZ')
    female=bool(top and top.get('gs_category')=='FEMALE_KAMEEZ')
    import male_eastern_finish
    stable_male=male_eastern_finish.owned(top)
    for obj,name in [(bottom,'AR Covered Lower Depth v4'),(scarf,'AR Dupatta Hem Limit v4')]:
        if obj and obj.modifiers.get(name):
            active=bool(enabled) and not ((female or stable_male) and name=='AR Covered Lower Depth v4')
            obj.modifiers[name].show_viewport=active;obj.modifiers[name].show_render=active
    if not enabled:return states
    group=make_group('AR Covered Lower Depth v4',[
        ('Upper','NodeSocketObject','INPUT'),('Ray','NodeSocketVector','INPUT'),('Clearance','NodeSocketFloat','INPUT')])
    if not group.nodes:
        node,c,m,v=helpers(group);inp=node('NodeGroupInput');out=node('NodeGroupOutput')
        obj=node('GeometryNodeObjectInfo');obj.transform_space='RELATIVE';c(inp.outputs['Upper'],obj.inputs['Object'])
        pos=node('GeometryNodeInputPosition');ray=node('GeometryNodeRaycast');ray.data_type='FLOAT'
        c(obj.outputs['Geometry'],ray.inputs['Target Geometry'])
        origin=v('SUBTRACT',pos.outputs[0],v('SCALE',inp.outputs['Ray'],1000.))
        c(origin,ray.inputs['Source Position']);c(inp.outputs['Ray'],ray.inputs['Ray Direction']);ray.inputs['Ray Length'].default_value=2000.
        behind=v('DOT_PRODUCT',v('SUBTRACT',pos.outputs[0],ray.outputs['Hit Position']),inp.outputs['Ray'])
        delta=m('MAXIMUM',m('SUBTRACT',inp.outputs['Clearance'],behind),0.)
        setpos=node('GeometryNodeSetPosition');c(inp.outputs['Geometry'],setpos.inputs['Geometry'])
        c(ray.outputs['Is Hit'],setpos.inputs['Selection']);c(v('SCALE',inp.outputs['Ray'],delta),setpos.inputs['Offset'])
        c(setpos.outputs[0],out.inputs['Geometry'])
    if not (female or stable_male):
        state=modifier(bottom,group);state[1][state[2]['Upper']]=top;states.append(state)
    if scarf and scarf.get('gs_category')=='FEMALE_DUPATTA':
        group=make_group('AR Dupatta Hem Limit v4',[
            ('Lower','NodeSocketObject','INPUT'),('Height row','NodeSocketVector','INPUT'),('Up','NodeSocketVector','INPUT')])
        if not group.nodes:
            node,c,m,v=helpers(group);inp=node('NodeGroupInput');out=node('NodeGroupOutput')
            obj=node('GeometryNodeObjectInfo');obj.transform_space='RELATIVE';c(inp.outputs['Lower'],obj.inputs['Object'])
            pos=node('GeometryNodeInputPosition');height=v('DOT_PRODUCT',pos.outputs[0],inp.outputs['Height row'])
            own=node('GeometryNodeAttributeStatistic');own.data_type='FLOAT';own.domain='POINT'
            low=node('GeometryNodeAttributeStatistic');low.data_type='FLOAT';low.domain='POINT'
            c(inp.outputs['Geometry'],own.inputs['Geometry']);c(height,own.inputs['Attribute'])
            c(obj.outputs['Geometry'],low.inputs['Geometry']);c(height,low.inputs['Attribute'])
            # Only shorten, with a continuous taper below the shoulder. Do not
            # pile every long vertex into a flattened, clipped bottom row.
            span=m('MAXIMUM',m('SUBTRACT',own.outputs['Max'],own.outputs['Min']),.001)
            anchor=m('SUBTRACT',own.outputs['Max'],m('MULTIPLY',span,.15))
            t=m('DIVIDE',m('SUBTRACT',anchor,height),m('MAXIMUM',m('SUBTRACT',anchor,own.outputs['Min']),.001))
            t=m('MINIMUM',m('MAXIMUM',t,0.),1.)
            shortening=m('MAXIMUM',m('SUBTRACT',m('ADD',low.outputs['Min'],.003),own.outputs['Min']),0.)
            setpos=node('GeometryNodeSetPosition');c(inp.outputs['Geometry'],setpos.inputs['Geometry'])
            c(v('SCALE',inp.outputs['Up'],m('MULTIPLY',t,shortening)),setpos.inputs['Offset'])
            c(setpos.outputs[0],out.inputs['Geometry'])
        state=modifier(scarf,group);state[1][state[2]['Lower']]=bottom;states.append(state)
    update(states)
    return states

def update(states):
    for obj,mod,sockets in states:
        matrix=obj.parent.matrix_world@obj.matrix_parent_inverse@obj.matrix_basis if obj.parent else obj.matrix_world
        inv=matrix.inverted().to_3x3()
        if 'Ray' in sockets:
            direction=inv@Vector((0,1,0));direction.normalize()
            values={'Ray':tuple(direction),'Clearance':float((inv@Vector((0,.008,0))).length)}
        else:values={'Height row':tuple(matrix[2][:3]),'Up':tuple(inv@Vector((0,0,1)))}
        changed=False
        for key,value in values.items():
            old=mod.get(sockets[key])
            different=old is None or (any(abs(a-b)>1e-7 for a,b in zip(old,value)) if isinstance(value,tuple) else abs(old-value)>1e-7)
            if different:mod[sockets[key]]=value;changed=True
        if changed:obj.update_tag()
