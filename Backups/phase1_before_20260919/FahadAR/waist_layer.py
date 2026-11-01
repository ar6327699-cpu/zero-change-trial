"""Studio-only untucked shirt layering, evaluated after tracking deformation.

Keep overlapping waistband vertices inside the current shirt. Original mesh,
UVs, rig weights and calibration are not rewritten. Not a cloth simulation.
"""
import numpy as np

NAME='AR Waist Under Shirt'
GROUP='AR Waist Overlap'

def basis_coords(obj):
    data=obj.data.shape_keys.key_blocks[0].data if obj.data.shape_keys else obj.data.vertices
    return np.array([v.co[:] for v in data],dtype=float)

def hem_profile(obj):
    c=basis_coords(obj)
    # Boundary loops distinguish the actual curved hem from lower folds.
    counts={}
    for p in obj.data.polygons:
        ids=list(p.vertices)
        for a,b in zip(ids,ids[1:]+ids[:1]):
            k=tuple(sorted((a,b)));counts[k]=counts.get(k,0)+1
    boundary=sorted({i for edge,n in counts.items() if n==1 for i in edge})
    cutoff=c[:,1].min()+np.ptp(c[:,1])*.28
    hem=c[boundary];hem=hem[hem[:,1]<cutoff]
    if len(hem)<8:
        # The shipped Mixamo garment has a closed/thickened hem. Recover its
        # lower envelope around the torso instead of requiring open topology.
        lower=c[c[:,1]<cutoff];centre=np.median(lower[:,[0,2]],axis=0)
        angles=np.arctan2(lower[:,2]-centre[1],lower[:,0]-centre[0])
        bins=np.floor((angles+np.pi)/(2*np.pi)*64).astype(int)
        hem=np.array([lower[np.flatnonzero(bins==k)[np.argmin(lower[bins==k,1])]] for k in np.unique(bins)])
    if len(hem)<8:raise ValueError('Shirt lower hem is unclear; inspect the template before layering')
    return c,hem

def ensure(arm):
    import bpy
    # Never attach this rule in the known-good original AR project.
    if not any(s.get('GS_studio_version') for s in bpy.data.scenes):return
    top=bpy.data.objects.get('Tops');bottom=bpy.data.objects.get('Bottoms')
    if top is None or bottom is None:return
    c,hem=hem_profile(top);b=basis_coords(bottom)
    # Templates share the same rig-local coordinates. Fail safely if a new
    # importer supplies a different object transform instead of misfitting it.
    if not np.allclose(np.array(top.matrix_basis),np.array(bottom.matrix_basis),atol=1e-5) or not np.allclose(np.array(top.matrix_parent_inverse),np.array(bottom.matrix_parent_inverse),atol=1e-5):
        raise ValueError('Garment transforms differ; normalize the new template before layering')
    centre=np.median(hem[:,[0,2]],axis=0)
    ha=np.arctan2(hem[:,2]-centre[1],hem[:,0]-centre[0])
    ba=np.arctan2(b[:,2]-centre[1],b[:,0]-centre[0])
    distance=np.abs(np.arctan2(np.sin(ba[:,None]-ha),np.cos(ba[:,None]-ha)))
    nearest=np.argsort(distance,axis=1)[:,:4]
    level=np.median(hem[nearest,1],axis=1)
    # Full containment starts slightly below the hem; a short smooth taper
    # underneath it avoids an abrupt waistband ridge. Legs get exactly zero.
    t=np.clip((b[:,1]-level+10)/8,0,1);weights=t*t*(3-2*t)
    vg=bottom.vertex_groups.get(GROUP) or bottom.vertex_groups.new(name=GROUP)
    vg.remove(list(range(len(b))))
    for i,w in enumerate(weights):
        if w>0:vg.add([i],float(w),'REPLACE')
    mod=bottom.modifiers.get(NAME) or bottom.modifiers.new(NAME,'SHRINKWRAP')
    mod.target=top;mod.wrap_method='NEAREST_SURFACEPOINT';mod.wrap_mode='INSIDE'
    mod.offset=1.2;mod.vertex_group=GROUP
    # Activation rebuilds the landmark modifier. Layering must always be last.
    index=list(bottom.modifiers).index(mod)
    bottom.modifiers.move(index,len(bottom.modifiers)-1)
    mod.show_viewport=True;mod.show_render=True
    bottom['gs_waist_layer']='Untucked: waistband inside current shirt; post-fit'
    bottom['gs_waist_layer_vertices']=int(np.count_nonzero(weights))
    return {'affected_vertices':int(np.count_nonzero(weights)),
            'fully_contained_vertices':int(np.count_nonzero(weights>.999)),
            'clearance_rig_units':1.2,'hem_min':float(hem[:,1].min()),'hem_max':float(hem[:,1].max())}
