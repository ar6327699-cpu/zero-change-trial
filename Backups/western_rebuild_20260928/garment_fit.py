"""AR-copy-only post-skinning landmark corrections, evaluated natively by Blender.

Screen-space visual fitting, not inferred body circumferences. Static smooth
weights blend observed joint residuals without rewriting mesh vertices per frame.
"""
import numpy as np

CONTROL_BONES={11:'LeftArm',12:'RightArm',23:'LeftUpLeg',24:'RightUpLeg',
               13:'LeftForeArm',14:'RightForeArm',25:'LeftLeg',26:'RightLeg',
               27:'LeftFoot',28:'RightFoot',15:'LeftHand',16:'RightHand'}
REGIONS={'Tops':[11,12,23,24,13,14],'Bottoms':[23,24,25,26],'Shoes':[27,28],
         'Dupatta':[11,12,23,24,13,14,15,16,25,26]}

def region_weights(coords,rest,name,female=False):
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
    if female:
        # A continuous pelvis belt must not receive opposite hip corrections
        # across a 10-unit strip. Blend across the waist and progressively
        # separate legs only below the crotch, where topology is separate.
        half=np.interp(y,[110,165,195],[5,14,32])
        side=np.clip((x+half)/(2*half),0,1)
    else:side=np.clip((x+5)/10,0,1)
    side=side*side*(3-2*side)
    return np.c_[(1-lower)*side,(1-lower)*(1-side),lower*side,lower*(1-side)]

def screen_residuals(p,projected,torso):
    p=np.asarray(p,float)
    if p.shape!=(33,4) or not np.isfinite(p).all():return {}
    result={}
    for i,point in projected.items():
        if p[i,3]<.55 or not (.005<p[i,0]<.995 and .005<p[i,1]<.995):continue
        delta=np.array([(p[i,0]-.5)*(32/9),(.5-p[i,1])*2])-point[[0,2]]
        # Reject mistracked joints instead of dragging cloth across the body.
        # Limb proportions differ from the avatar. A knee/ankle displaced by
        # half a torso is a valid fit correction, not automatically an outlier.
        limit=torso*(1.25 if i in (13,14,15,16,25,26,27,28) else .55)
        if np.linalg.norm(delta)>max(.08,limit):continue
        result[i]=np.array([delta[0],0.,delta[1]])
    return result

class GarmentFitter:
    def __init__(self,arm):
        import bpy
        import western_fit
        self.arm=arm;self.items={};self.hems={};self.residual_history={};self.rest={i:np.array(arm.data.bones['mixamorig:'+n].head_local) for i,n in CONTROL_BONES.items()}
        self.female=bpy.data.objects.get('Tops') is not None and bpy.data.objects['Tops'].get('gs_category')=='FEMALE_KAMEEZ'
        from female_stability import ResidualLock
        self.female_lock=ResidualLock()
        # Layer binding must precede residual ownership and dupatta transfer.
        # Previously those fields were built from last outfit's leg weights.
        if self.female:
            import waist_layer
            waist_layer.ensure(arm)
        for name,base_ids in REGIONS.items():
            obj=bpy.data.objects.get(name)
            if obj is None:continue
            category=obj.get('gs_category','')
            ids=list(base_ids)
            basis=obj.data.shape_keys.key_blocks[0].data if obj.data.shape_keys else obj.data.vertices
            coords=np.array([v.co[:] for v in basis])
            if name=='Dupatta':
                top=bpy.data.objects.get('Tops')
                if top is None or top.get('gs_category')!='FEMALE_KAMEEZ':continue
                import accessories
                weights,ids=accessories.attachment_weights(obj,top)
            else:weights=region_weights(coords,self.rest,name,category.startswith('FEMALE_'))
            if category in ('FULL_SHIRT','KURTA','SHALWAR','PAJAMA','FEMALE_KAMEEZ','FEMALE_TROUSER','FEMALE_SHALWAR','FEMALE_DUPATTA'):
                for modifier in obj.modifiers:
                    if modifier.type=='ARMATURE':modifier.use_deform_preserve_volume=not category.startswith('FEMALE_')
            if name=='Tops' and (category in ('FEMALE_KAMEEZ','KURTA') or obj.get('gs_id') in western_fit.TOPS):
                # A sleeve's lower circumference is still attached to an arm,
                # not to the hips merely because its REST y is lower. Follow
                # the authored shoulder seam and interpolate along the sleeve.
                sleeve=np.zeros_like(weights);amount=np.zeros(len(coords))
                for side,shoulder,elbow,column in [('Left',11,13,0),('Right',12,14,1)]:
                    group_ids={g.index for g in obj.vertex_groups if g.name in
                        ('mixamorig:'+side+'Arm','mixamorig:'+side+'ForeArm','mixamorig:'+side+'Hand')}
                    attach=np.array([sum(g.weight for g in v.groups if g.group in group_ids) for v in obj.data.vertices])
                    t=np.clip((np.abs(coords[:,0])-abs(self.rest[shoulder][0]))/
                              max(abs(self.rest[elbow][0])-abs(self.rest[shoulder][0]),1),0,1)
                    sleeve[:,column]=attach*(1-t);sleeve[:,column+4]=attach*t
                    amount+=attach
                weights=weights*(1-np.clip(amount,0,1))[:,None]+sleeve
            if name=='Bottoms' and category.startswith('FEMALE_'):
                # Wide shalwar legs may cross x=0 in REST space. Side ownership
                # comes from skin weights, not the vertex's global x sign.
                left_ids={g.index for g in obj.vertex_groups if g.name in ('mixamorig:LeftUpLeg','mixamorig:LeftLeg','mixamorig:LeftFoot')}
                right_ids={g.index for g in obj.vertex_groups if g.name in ('mixamorig:RightUpLeg','mixamorig:RightLeg','mixamorig:RightFoot')}
                ownership=[]
                for v in obj.data.vertices:
                    left=sum(g.weight for g in v.groups if g.group in left_ids)
                    right=sum(g.weight for g in v.groups if g.group in right_ids)
                    # Keep the pelvis contribution: normalizing it away turns
                    # a mostly-hip vertex into a full left/right correction.
                    pelvis=max(0.,1-left-right)
                    j=v.index
                    ownership.append(left+pelvis*(weights[j,0]+weights[j,2]))
                side=weights[:,0]+weights[:,2]
                known=np.isfinite(ownership);side[known]=np.asarray(ownership)[known]
                proximal=weights[:,0]+weights[:,1]
                weights=np.c_[proximal*side,proximal*(1-side),(1-proximal)*side,(1-proximal)*(1-side)]
            if name=='Tops' and category in ('FULL_SHIRT','KURTA','FEMALE_KAMEEZ'):
                extra=[]
                for column,elbow,wrist in [(4,13,15),(5,14,16)]:
                    t=np.clip((np.abs(coords[:,0])-abs(self.rest[elbow][0]))/max(abs(self.rest[wrist][0])-abs(self.rest[elbow][0]),1),0,1)
                    extra.append(weights[:,column]*t);weights[:,column]*=1-t
                weights=np.c_[weights,*extra];ids += [15,16]
            if name=='Tops' and category in ('KURTA','FEMALE_KAMEEZ'):
                from eastern_foundation import extend_kurta_fit
                weights,ids=extend_kurta_fit(weights,coords,self.rest,ids,strength=0.)
            if name=='Bottoms' and category in ('PANTS','TROUSERS','SHALWAR','PAJAMA','FEMALE_TROUSER','FEMALE_SHALWAR'):
                extra=[]
                for column,knee,ankle in [(2,25,27),(3,26,28)]:
                    t=np.clip((self.rest[knee][1]-coords[:,1])/max(self.rest[knee][1]-self.rest[ankle][1],1),0,1)
                    extra.append(weights[:,column]*t);weights[:,column]*=1-t
                weights=np.c_[weights,*extra];ids += [27,28]
            if name=='Bottoms' and category in ('SHALWAR','PAJAMA','FEMALE_TROUSER','FEMALE_SHALWAR'):
                top=bpy.data.objects.get('Tops')
                if top and top.get('gs_category') in ('KURTA','FEMALE_KAMEEZ'):
                    top_basis=top.data.shape_keys.key_blocks[0].data if top.data.shape_keys else top.data.vertices
                    hem=min(v.co.y for v in top_basis);t=np.clip((coords[:,1]-145)/50,0,1);t=t*t*(3-2*t)
                    hip=(self.rest[23]+self.rest[24])*.5
                    side=np.clip((coords[:,0]+abs(self.rest[23][0]))/(2*abs(self.rest[23][0])),0,1)
                    shared=np.zeros_like(weights);shared[:,0]=side;shared[:,1]=1-side
                    weights=weights*(1-t[:,None])+shared*t[:,None]
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
            if category=='KURTA':
                import eastern_live
                self.hems[name]=eastern_live.knee_modifier(obj,self.rest,coords)
        self.reset()
        # Recompute against whichever shirt and bottoms are active, including
        # newly imported garments. Native evaluation follows both live poses.
        import waist_layer
        waist_layer.ensure(arm)
        import accessories
        for obj,mod,sockets in self.items.values():accessories.surface(obj)
        import garment_constraints
        self.constraints=garment_constraints.ensure()
    def reset(self):
        self.residual_history={}
        self.female_lock.reset()
        for obj,mod,sockets in self.items.values():
            for key in sockets.values():mod[key]=(0.,0.,0.)
            obj.update_tag()
        for state in self.hems.values():
            if state:state[0][state[1]['Enabled']]=0.
    def update(self,p,source,fit,stamp=None):
        from mathutils import Vector
        import garment_constraints
        garment_constraints.update(self.constraints)
        sx,sz,tx,tz=fit
        projected={i:v*np.array([sx,np.sqrt(sx*sz),sz])+[tx,0,tz] for i,v in source.items()}
        torso=np.mean([np.linalg.norm(projected[a][[0,2]]-projected[b][[0,2]]) for a,b in [(11,23),(12,24)]])
        deltas=screen_residuals(p,projected,torso)
        measured_count=len(deltas)
        if self.female:
            deltas=self.female_lock.update(deltas,stamp if stamp is not None else 0.)
        elif any(obj.get('gs_category') in ('KURTA','FEMALE_KAMEEZ') for obj,mod,sockets in self.items.values()):
            # Bound corrections relative to the torso; the rig handles gross
            # movement. Residual jitter must not refit the whole garment.
            for i,value in list(deltas.items()):
                norm=np.linalg.norm(value)
                limit=max(.02,torso*(1.25 if i in (13,14,15,16,25,26,27,28) else .55))
                if norm>limit:value=value*limit/norm
                old=self.residual_history.get(i)
                if old is not None:
                    distance=np.linalg.norm(value-old)
                    # The input landmarks already have temporal filtering.
                    # Only damp subpixel residual noise, not real limb travel.
                    value=old if distance<.001 else old+(value-old)*(.98 if distance>.012 else .65)
                deltas[i]=value;self.residual_history[i]=value.copy()
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
            if name in self.hems:
                import eastern_live
                eastern_live.update_hem(obj,self.hems[name],matrix,np.asarray(p,float),projected)
        return {'corrected_joints':measured_count,'shoulder_correction_px':float(np.mean([np.linalg.norm(deltas[i])*180 for i in [11,12] if i in deltas])) if any(i in deltas for i in [11,12]) else None}
