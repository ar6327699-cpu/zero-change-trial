"""AR-copy-only post-skinning landmark corrections, evaluated natively by Blender.

Screen-space visual fitting, not inferred body circumferences. Static smooth
weights blend observed joint residuals without rewriting mesh vertices per frame.
"""
import numpy as np

CONTROL_BONES={11:'LeftArm',12:'RightArm',23:'LeftUpLeg',24:'RightUpLeg',
               13:'LeftForeArm',14:'RightForeArm',25:'LeftLeg',26:'RightLeg',
               27:'LeftFoot',28:'RightFoot',15:'LeftHand',16:'RightHand'}
REGIONS={'Tops':[11,12,23,24,13,14],'Bottoms':[23,24,25,26],'Shoes':[27,28]}

def region_weights(coords,rest,name):
    x,y=np.asarray(coords)[:,:2].T
    if name=='Shoes':
        left=np.clip((x+2)/4,0,1);return np.c_[left,1-left]
    hip=(rest[23][1]+rest[24][1])*.5
    if name=='Tops':
        sh=(abs(rest[11][0])+abs(rest[12][0]))*.5
        sy=(rest[11][1]+rest[12][1])*.5
        top=np.clip((y-hip)/(sy-hip),0,1)
        side=np.clip((x+sh)/(2*sh),0,1)
        hs=np.clip((x+abs(rest[23][0]))/(2*abs(rest[23][0])),0,1)
        elbow=(abs(rest[13][0])+abs(rest[14][0]))*.5
        sleeve=np.clip((abs(x)-sh)/max(elbow-sh,1),0,1)
        return np.c_[top*side*(1-sleeve),top*(1-side)*(1-sleeve),
                     (1-top)*hs,(1-top)*(1-hs),top*side*sleeve,top*(1-side)*sleeve]
    knee=(rest[25][1]+rest[26][1])*.5
    lower=np.clip((hip-y)/(hip-knee),0,1)
    # Narrow smooth crotch transition preserves independent left/right legs.
    side=np.clip((x+5)/10,0,1);side=side*side*(3-2*side)
    return np.c_[(1-lower)*side,(1-lower)*(1-side),lower*side,lower*(1-side)]

def screen_residuals(p,projected,torso):
    p=np.asarray(p,float)
    if p.shape!=(33,4) or not np.isfinite(p).all():return {}
    result={}
    for i,point in projected.items():
        if p[i,3]<.55 or not (.005<p[i,0]<.995 and .005<p[i,1]<.995):continue
        delta=np.array([(p[i,0]-.5)*(32/9),(.5-p[i,1])*2])-point[[0,2]]
        # Reject mistracked joints instead of dragging cloth across the body.
        if np.linalg.norm(delta)>max(.08,torso*.38):continue
        result[i]=np.array([delta[0],0.,delta[1]])
    return result

class GarmentFitter:
    def __init__(self,arm):
        import bpy
        self.arm=arm;self.items={};self.rest={i:np.array(arm.data.bones['mixamorig:'+n].head_local) for i,n in CONTROL_BONES.items()}
        for name,base_ids in REGIONS.items():
            obj=bpy.data.objects.get(name)
            if obj is None:continue
            category=obj.get('gs_category','')
            ids=list(base_ids)
            basis=obj.data.shape_keys.key_blocks[0].data if obj.data.shape_keys else obj.data.vertices
            coords=np.array([v.co[:] for v in basis]);weights=region_weights(coords,self.rest,name)
            if name=='Tops' and category in ('FULL_SHIRT','KURTA'):
                extra=[]
                for column,elbow,wrist in [(4,13,15),(5,14,16)]:
                    t=np.clip((np.abs(coords[:,0])-abs(self.rest[elbow][0]))/max(abs(self.rest[wrist][0])-abs(self.rest[elbow][0]),1),0,1)
                    extra.append(weights[:,column]*t);weights[:,column]*=1-t
                weights=np.c_[weights,*extra];ids += [15,16]
            if name=='Tops' and category=='KURTA':
                from eastern_foundation import extend_kurta_fit
                weights,ids=extend_kurta_fit(weights,coords,self.rest,ids)
            if name=='Bottoms' and category in ('PANTS','TROUSERS','SHALWAR'):
                extra=[]
                for column,knee,ankle in [(2,25,27),(3,26,28)]:
                    t=np.clip((self.rest[knee][1]-coords[:,1])/max(self.rest[knee][1]-self.rest[ankle][1],1),0,1)
                    extra.append(weights[:,column]*t);weights[:,column]*=1-t
                weights=np.c_[weights,*extra];ids += [27,28]
            assert np.allclose(weights.sum(1),1) and weights.min()>=0
            group_name='AR Fit '+name+(' '+category if len(ids)!=len(base_ids) else '')
            group=bpy.data.node_groups.get(group_name)
            if group is None:
                group=bpy.data.node_groups.new(group_name,'GeometryNodeTree')
                group.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
                group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
                for i in ids:group.interface.new_socket(name='Joint '+str(i),in_out='INPUT',socket_type='NodeSocketVector')
                nodes=group.nodes;links=group.links
                inp=nodes.new('NodeGroupInput');out=nodes.new('NodeGroupOutput');setpos=nodes.new('GeometryNodeSetPosition')
                links.new(inp.outputs['Geometry'],setpos.inputs['Geometry']);links.new(setpos.outputs['Geometry'],out.inputs['Geometry'])
                total=None
                for i in ids:
                    attr=nodes.new('GeometryNodeInputNamedAttribute');attr.data_type='FLOAT';attr.inputs['Name'].default_value='AR_fit_'+str(i)
                    scale=nodes.new('ShaderNodeVectorMath');scale.operation='SCALE'
                    links.new(inp.outputs['Joint '+str(i)],scale.inputs[0]);links.new(attr.outputs['Attribute'],scale.inputs['Scale'])
                    if total is None:total=scale.outputs['Vector']
                    else:
                        add=nodes.new('ShaderNodeVectorMath');add.operation='ADD'
                        links.new(total,add.inputs[0]);links.new(scale.outputs['Vector'],add.inputs[1]);total=add.outputs['Vector']
                links.new(total,setpos.inputs['Offset'])
            for k,i in enumerate(ids):
                attr=obj.data.attributes.get('AR_fit_'+str(i)) or obj.data.attributes.new('AR_fit_'+str(i),'FLOAT','POINT')
                attr.data.foreach_set('value',weights[:,k].astype(np.float32))
            mod=obj.modifiers.get('AR Landmark Fit') or obj.modifiers.new('AR Landmark Fit','NODES');mod.node_group=group
            sockets={int(v.name.split()[-1]):v.identifier for v in group.interface.items_tree if v.item_type=='SOCKET' and v.in_out=='INPUT' and v.name.startswith('Joint ')}
            self.items[name]=(obj,mod,sockets)
        self.reset()
        # Recompute against whichever shirt and bottoms are active, including
        # newly imported garments. Native evaluation follows both live poses.
        import waist_layer
        waist_layer.ensure(arm)
    def reset(self):
        for obj,mod,sockets in self.items.values():
            for key in sockets.values():mod[key]=(0.,0.,0.)
            obj.update_tag()
    def update(self,p,source,fit):
        from mathutils import Vector
        sx,sz,tx,tz=fit
        projected={i:v*np.array([sx,np.sqrt(sx*sz),sz])+[tx,0,tz] for i,v in source.items()}
        torso=np.mean([np.linalg.norm(projected[a][[0,2]]-projected[b][[0,2]]) for a,b in [(11,23),(12,24)]])
        deltas=screen_residuals(p,projected,torso)
        for name,(obj,mod,sockets) in self.items.items():
            # Parent transform is computed directly, avoiding another full
            # depsgraph evaluation after global projection changes.
            matrix=self.arm.matrix_world@obj.matrix_parent_inverse@obj.matrix_basis
            inverse=matrix.inverted().to_3x3();changed=False
            for i,key in sockets.items():
                delta=deltas.get(i)
                if delta is None:
                    # Missing elbow/knee follows its same-side proximal joint;
                    # missing far torso follows near torso, not last-frame data.
                    fallback={13:11,14:12,15:13,16:14,25:23,26:24,11:12,12:11,23:24,24:23,27:25,28:26}.get(i)
                    delta=deltas.get(fallback,np.zeros(3))
                local=tuple(inverse@Vector(delta))
                if np.max(np.abs(np.array(mod[key])-local))>1e-5:mod[key]=local;changed=True
            if changed:obj.update_tag()
        return {'corrected_joints':len(deltas),'shoulder_correction_px':float(np.mean([np.linalg.norm(deltas[i])*180 for i in [11,12] if i in deltas])) if any(i in deltas for i in [11,12]) else None}
