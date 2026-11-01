"""Current visible clothing envelope for female try-on; no body-size inference.

Only numeric outline samples cross the camera socket. The native modifier
expands the existing textured cloth to cover supported visible edges. Source
vertices, UVs, saved fitting controls, and the camera image are unchanged.
"""
import numpy as np

def side_lining(top,base):
    """Cover the authored side vents with the same cloth during live try-on.

    Reuse existing boundary indices in a separate runtime mesh. The native
    node graph samples their deformed positions; no new tracking or weights.
    """
    import bpy
    edges={}
    for face in top.data.polygons:
        ids=list(face.vertices)
        for a,b in zip(ids,ids[1:]+ids[:1]):
            key=tuple(sorted((a,b)));edges[key]=edges.get(key,0)+1
    panels=[]
    for sign in (1,-1):
        boundary=set()
        for (a,b),count in edges.items():
            if count==1 and abs(base[a,1]-base[b,1])>1 and all(base[i,0]*sign>30 and base[i,1]<=200 for i in (a,b)):
                boundary.update((a,b))
        rows=[]
        for y in sorted(set(round(base[i,1],3) for i in boundary)):
            ids=[i for i in boundary if abs(base[i,1]-y)<.002]
            if len(ids)!=2:continue
            ids.sort(key=lambda i:base[i,2])
            middle=np.flatnonzero((abs(base[:,1]-y)<.002)&(base[:,0]*sign>30)&
                (base[:,2]>base[ids[0],2]+.001)&(base[:,2]<base[ids[1],2]-.001))
            if len(middle):ids.insert(1,int(middle[np.argmax(base[middle,0]*sign)]))
            rows.append(ids)
        for a,b in zip(rows,rows[1:]):
            if len(a)!=len(b):continue
            for j in range(len(a)-1):panels.append((a[j],a[j+1],b[j+1],b[j]))
    if not panels:return None
    used=sorted(set(i for face in panels for i in face));mapping={j:i for i,j in enumerate(used)}
    mesh=bpy.data.meshes.new('AR Female Side Lining Source')
    mesh.from_pydata(base[used].tolist(),[],[tuple(mapping[i] for i in face) for face in panels]);mesh.update()
    attr=mesh.attributes.new('AR_side_source','INT','POINT');attr.data.foreach_set('value',used)
    attr=mesh.attributes.new('AR_source_index','INT','POINT');attr.data.foreach_set('value',used)
    seam=top.data.attributes.get('AR_vent_boundary') or top.data.attributes.new('AR_vent_boundary','BOOLEAN','POINT')
    selected=np.zeros(len(base),bool);selected[used]=True;seam.data.foreach_set('value',selected)
    seam=mesh.attributes.new('AR_vent_boundary','BOOLEAN','POINT');seam.data.foreach_set('value',np.ones(len(used),bool))
    for material in top.data.materials:mesh.materials.append(material)
    # Side shaders already blend the photo edge into their sampled cloth
    # colour. Use that same side finish instead of inventing new print.
    blend=mesh.color_attributes.new(name='GS_side_blend',type='FLOAT_COLOR',domain='CORNER')
    blend.data.foreach_set('color',np.ones(len(mesh.loops)*4,np.float32))
    for source_uv in top.data.uv_layers:
        uv=mesh.uv_layers.new(name=source_uv.name);by_vertex={}
        for loop in top.data.loops:by_vertex.setdefault(loop.vertex_index,tuple(source_uv.data[loop.index].uv))
        for loop in mesh.loops:uv.data[loop.index].uv=by_vertex.get(used[loop.vertex_index],(.5,.5))
    for face in mesh.polygons:face.use_smooth=True
    obj=bpy.data.objects.new('AR Female Side Lining Source',mesh)
    # Unlinked helper: available to Object Info but never directly drawn.
    return obj

def measure(mask,pose,stamp):
    p=np.asarray(pose,float);out={'stamp':float(stamp),'rows':[],'sleeves':{}}
    if mask is None or p.shape!=(33,4) or not np.isfinite(p).all():return out
    import cv2
    a=np.asarray(mask,float)
    if a.ndim==3:a=a[:,:,0]
    if a.ndim!=2 or not np.isfinite(a).all():return out
    a=cv2.resize(a,(640,360))>.60
    xy=p[:,:2]*[640,360]
    seen=lambda j:p[j,3]>=.60 and np.all(p[j,:2]>0) and np.all(p[j,:2]<1)
    pairs=[(s,h) for s,h in [(11,23),(12,24)] if seen(s) and seen(h)]
    if not pairs:return out
    shoulder=np.mean([xy[s] for s,h in pairs],axis=0)
    hip=np.mean([xy[h] for s,h in pairs],axis=0)
    height=hip[1]-shoulder[1]
    if height<25:return out
    out['torso_px']=float(height)
    for t in (.10,.22,.40,.62,.84,1.05,1.30,1.55,1.80):
        y=int(round(shoulder[1]+height*t));cx=shoulder[0]+(hip[0]-shoulder[0])*min(t,1)
        if not 2<=y<358:continue
        line=np.mean(a[y-1:y+2],axis=0)>.5
        original_seed=int(np.clip(round(cx),0,639))
        if line[original_seed]:
            edge_lo=edge_hi=original_seed
            while edge_lo>0 and line[edge_lo-1]:edge_lo-=1
            while edge_hi<639 and line[edge_hi+1]:edge_hi+=1
            if edge_lo<2 or edge_hi>637:continue
        # An arm touching the torso joins its silhouette into the same mask
        # component. Split clearly lateral sleeve corridors before measuring
        # the trunk, so lowering an arm cannot inflate the whole chest.
        for j,k in [(11,13),(13,15),(12,14),(14,16)]:
            if not seen(j) or not seen(k):continue
            dy=xy[k,1]-xy[j,1]
            if abs(dy)<3:continue
            t_arm=(y-xy[j,1])/dy
            if not 0<=t_arm<=1:continue
            arm_x=xy[j,0]+t_arm*(xy[k,0]-xy[j,0])
            if abs(arm_x-cx)>height*.28:
                radius=height*.095
                line[max(0,int(arm_x-radius)):min(640,int(arm_x+radius)+1)]=False
        search=np.arange(max(0,int(cx-height*.12)),min(640,int(cx+height*.12)+1))
        hits=search[line[search]]
        if not len(hits):continue
        seed=int(hits[np.argmin(abs(hits-cx))]);lo=hi=seed
        while lo>0 and line[lo-1]:lo-=1
        while hi<639 and line[hi+1]:hi+=1
        width=hi-lo
        # Reject an arm spanning the row, a background bridge, or a clipped
        # silhouette. Unknown rows do not become an invented coverage target.
        if lo<2 or hi>637 or not height*.18<width<height*1.45:continue
        if max(cx-lo,hi-cx)>height*.9:continue
        out['rows'].append([float(y),float(lo-1.5),float(hi+1.5)])
    for side,s,e,w in [('left',11,13,15),('right',12,14,16)]:
        samples=[]
        for j,k in [(s,e),(e,w)]:
            if not seen(j) or not seen(k):samples.append(None);continue
            delta=xy[k]-xy[j];length=np.linalg.norm(delta)
            if length<8:samples.append(None);continue
            normal=np.array([-delta[1],delta[0]])/length;radii=[]
            for fraction in (.35,.60,.80):
                centre=xy[j]+fraction*delta
                body_x=shoulder[0]+np.clip((centre[1]-shoulder[1])/height,0,1)*(hip[0]-shoulder[0])
                if shoulder[1]<centre[1]<hip[1]+height*.1 and abs(centre[0]-body_x)<height*.30:continue
                limit=int(np.clip(height*.16,4,20));d=np.arange(-limit,limit+1)
                points=np.rint(centre+d[:,None]*normal).astype(int)
                valid=(points[:,0]>=0)&(points[:,0]<640)&(points[:,1]>=0)&(points[:,1]<360)
                values=np.zeros(len(d),bool);values[valid]=a[points[valid,1],points[valid,0]]
                if not values[limit]:continue
                lo=hi=limit
                while lo>0 and values[lo-1]:lo-=1
                while hi<len(d)-1 and values[hi+1]:hi+=1
                if lo==0 or hi==len(d)-1:continue
                radii.append(max(limit-lo,hi-limit)+1.)
            samples.append(float(np.median(radii)) if len(radii)>=2 else None)
        if any(v is not None for v in samples):out['sleeves'][side]=samples
    return out

class SurfaceFitter:
    def __init__(self,arm,top):
        import bpy
        self.arm=arm;self.top=top;self.state={};self.outline=None
        self.base=np.array([tuple(v.co) for v in top.data.vertices])
        self.weights={}
        for group in top.vertex_groups:
            if group.name not in arm.data.bones:continue
            weight=np.array([sum(g.weight for g in v.groups if g.group==group.index) for v in top.data.vertices])
            if weight.max()>0:self.weights[group.name]=weight
        self.rest={name:np.array(arm.data.bones[name].matrix_local.inverted()) for name in self.weights}
        self.arm_weight=sum((w for n,w in self.weights.items() if any(n.endswith(x) for x in ('Arm','ForeArm','Hand'))),np.zeros(len(self.base)))
        self.rings=[]
        for side,sgn in [('left',1),('right',-1)]:
            eligible=(self.base[:,0]*sgn>45)&(self.base[:,1]>275)&(self.arm_weight>.99)
            for x in np.unique(np.round(self.base[eligible,0],3)):
                ids=np.flatnonzero(eligible&(abs(self.base[:,0]-x)<.002))
                if len(ids)>=8:self.rings.append((side,ids))
        self.body=[]
        for y in np.unique(np.round(self.base[:,1],3)):
            if not 90<=y<=308:continue
            ids=np.flatnonzero((abs(self.base[:,1]-y)<.002)&(self.arm_weight<.8)&(abs(self.base[:,0])<55))
            if len(ids)>=20:self.body.append(ids)
        self.attr=top.data.attributes.get('AR_surface_offset') or top.data.attributes.new('AR_surface_offset','FLOAT_VECTOR','POINT')
        source_index=top.data.attributes.get('AR_source_index') or top.data.attributes.new('AR_source_index','INT','POINT')
        source_index.data.foreach_set('value',np.arange(len(self.base),dtype=np.int32))
        self.lining=side_lining(top,self.base)
        group=bpy.data.node_groups.get('AR Female Visible Coverage')
        if group is None:
            group=bpy.data.node_groups.new('AR Female Visible Coverage','GeometryNodeTree')
            for io in ('INPUT','OUTPUT'):group.interface.new_socket(name='Geometry',in_out=io,socket_type='NodeSocketGeometry')
            group.interface.new_socket(name='Side lining',in_out='INPUT',socket_type='NodeSocketObject')
            nodes=group.nodes;links=group.links
            inp=nodes.new('NodeGroupInput');out=nodes.new('NodeGroupOutput');move=nodes.new('GeometryNodeSetPosition')
            attr=nodes.new('GeometryNodeInputNamedAttribute');attr.data_type='FLOAT_VECTOR';attr.inputs['Name'].default_value='AR_surface_offset'
            links.new(inp.outputs['Geometry'],move.inputs['Geometry']);links.new(attr.outputs['Attribute'],move.inputs['Offset'])
            lining=nodes.new('GeometryNodeObjectInfo');links.new(inp.outputs['Side lining'],lining.inputs['Object'])
            index=nodes.new('GeometryNodeInputNamedAttribute');index.data_type='INT';index.inputs['Name'].default_value='AR_side_source'
            position=nodes.new('GeometryNodeInputPosition');sample=nodes.new('GeometryNodeSampleIndex');sample.data_type='FLOAT_VECTOR';sample.domain='POINT'
            links.new(move.outputs[0],sample.inputs['Geometry']);links.new(position.outputs[0],sample.inputs['Value']);links.new(index.outputs['Attribute'],sample.inputs['Index'])
            panel=nodes.new('GeometryNodeSetPosition');links.new(lining.outputs['Geometry'],panel.inputs['Geometry']);links.new(sample.outputs['Value'],panel.inputs['Position'])
            join=nodes.new('GeometryNodeJoinGeometry');links.new(move.outputs[0],join.inputs['Geometry']);links.new(panel.outputs[0],join.inputs['Geometry'])
            merge=nodes.new('GeometryNodeMergeByDistance');merge.inputs['Distance'].default_value=.0001
            boundary=nodes.new('GeometryNodeInputNamedAttribute');boundary.data_type='BOOLEAN';boundary.inputs['Name'].default_value='AR_vent_boundary'
            links.new(boundary.outputs['Attribute'],merge.inputs['Selection']);links.new(join.outputs[0],merge.inputs['Geometry']);links.new(merge.outputs[0],out.inputs['Geometry'])
        self.mod=top.modifiers.get(group.name) or top.modifiers.new(group.name,'NODES');self.mod.node_group=group
        self.mod.show_viewport=True;self.mod.show_render=True
        socket=next(s.identifier for s in group.interface.items_tree if s.item_type=='SOCKET' and s.in_out=='INPUT' and s.name=='Side lining')
        previous=self.mod.get(socket);self.mod[socket]=self.lining
        if previous and previous!=self.lining and previous.name.startswith('AR Female Side Lining Source') and previous.users==0:
            data=previous.data;bpy.data.objects.remove(previous)
            if data.users==0:bpy.data.meshes.remove(data)
        fit=top.modifiers.get('AR Landmark Fit')
        top.modifiers.move(list(top.modifiers).index(self.mod),list(top.modifiers).index(fit)+1)
        self.clear()

    def clear(self):
        self.attr.data.foreach_set('vector',np.zeros(len(self.base)*3,np.float32));self.top.update_tag();self.state={};self.outline=None

    def predicted(self,sockets,fitmod):
        """Linear skinning of the small control cage; no extra depsgraph pass."""
        arm=self.arm;top=self.top;matrices={}
        if any(b.constraints for b in arm.pose.bones):return None
        if any(m.type=='ARMATURE' and m.use_deform_preserve_volume for m in top.modifiers):return None
        def pose(b):
            if b.name not in matrices:
                kw={} if b.parent is None else dict(parent_matrix=pose(b.parent),parent_matrix_local=b.parent.bone.matrix_local)
                matrices[b.name]=b.bone.convert_local_to_pose(b.matrix_basis,b.bone.matrix_local,**kw)
            return matrices[b.name]
        coords=self.base.copy();keys=top.data.shape_keys
        if keys:
            coords=np.array([tuple(v.co) for v in keys.key_blocks[0].data])
            for key in list(keys.key_blocks)[1:]:
                if key.mute or abs(key.value)<1e-7:continue
                relative=np.array([tuple(v.co) for v in key.relative_key.data])
                coords+=key.value*(np.array([tuple(v.co) for v in key.data])-relative)
        c=np.array(top.matrix_parent_inverse@top.matrix_basis);ci=np.linalg.inv(c)
        points=np.c_[coords,np.ones(len(coords))];skinned=np.zeros((len(coords),3));total=np.zeros(len(coords))
        for name,weight in self.weights.items():
            transform=ci@np.array(pose(arm.pose.bones[name]))@self.rest[name]@c
            skinned+=(points@transform.T)[:,:3]*weight[:,None];total+=weight
        skinned+=coords*(1-total)[:,None]
        for j,key in sockets.items():
            attr=top.data.attributes.get('AR_fit_'+str(j))
            if attr:
                weights=np.empty(len(coords),np.float32);attr.data.foreach_get('value',weights)
                skinned+=weights[:,None]*np.asarray(fitmod[key])
        matrix=np.array(arm.matrix_world@top.matrix_parent_inverse@top.matrix_basis)
        return (np.c_[skinned,np.ones(len(coords))]@matrix.T)[:,:3],matrix

    def update(self,evidence,stamp,p,sockets,fitmod,held=()):
        self.state={}
        if not isinstance(evidence,dict) or abs(float(evidence.get('stamp',-1))-stamp)>.001:
            self.clear();return
        predicted=self.predicted(sockets,fitmod)
        if predicted is None:self.clear();return
        vertices,matrix=predicted;screen=np.c_[vertices[:,0]*180+320,180-vertices[:,2]*180]
        offsets=np.zeros((len(screen),2));height=float(evidence.get('torso_px',0))
        rows=np.asarray(evidence.get('rows',[]),float)
        if rows.ndim==2 and rows.shape[1]==3 and len(rows)>=3 and np.isfinite(rows).all() and height>25:
            # A fixed one-pixel deadband rejects segmentation shimmer. Larger
            # real outline changes use this frame immediately.
            if self.outline is not None and self.outline.shape==rows.shape and (
                np.max(abs(rows-self.outline))<1. or ('torso' in held and np.max(abs(rows[:,0]-self.outline[:,0]))<2.)):
                rows=self.outline.copy()
            else:self.outline=rows.copy()
            for ids in self.body:
                y=float(np.median(screen[ids,1]))
                if not rows[0,0]-height*.12<=y<=rows[-1,0]+height*.12:continue
                lo,hi=np.min(screen[ids,0]),np.max(screen[ids,0]);span=hi-lo
                if span<3:continue
                want_lo=np.interp(y,rows[:,0],rows[:,1]);want_hi=np.interp(y,rows[:,0],rows[:,2])
                dl=float(np.clip(want_lo-lo,-height*.30,0));dr=float(np.clip(want_hi-hi,0,height*.30))
                t=(screen[ids,0]-lo)/span;offsets[ids,0]=(1-t)*dl+t*dr
            self.state['body_rows']=len(rows)
        pose=np.asarray(p,float)
        if pose.shape==(33,4):
            xy=pose[:,:2]*[640,360]
            for side,ids in self.rings:
                samples=evidence.get('sleeves',{}).get(side)
                if not samples:continue
                s,e,w=(11,13,15) if side=='left' else (12,14,16)
                elbow_x=abs(self.arm.data.bones['mixamorig:'+('Left' if side=='left' else 'Right')+'ForeArm'].head_local.x)
                segment=0 if abs(self.base[ids,0].mean())<elbow_x else 1
                radius=samples[segment]
                if radius is None or not np.isfinite(radius) or not 2<=radius<=20:continue
                a,b=(s,e) if segment==0 else (e,w);direction=xy[b]-xy[a];length=np.linalg.norm(direction)
                if length<8:continue
                normal=np.array([-direction[1],direction[0]])/length;centre=screen[ids].mean(0)
                signed=(screen[ids]-centre)@normal;current=max(abs(signed))
                if current<1:continue
                expansion=np.clip(radius/current,1,1.40)-1
                offsets[ids]+=signed[:,None]*normal*expansion
        world=np.c_[offsets[:,0]/180,np.zeros(len(offsets)),-offsets[:,1]/180]
        local=world@np.linalg.inv(matrix[:3,:3]).T
        if not np.isfinite(local).all():self.clear();return
        self.attr.data.foreach_set('vector',local.astype(np.float32).ravel());self.top.update_tag()
        self.state['max_expansion_px']=float(np.linalg.norm(offsets,axis=1).max())
