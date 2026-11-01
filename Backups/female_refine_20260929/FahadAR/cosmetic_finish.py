"""Scoped cosmetic corrections; no pose, fit, size or drape simulation changes."""
import bpy
import numpy as np

MEN={'7125317243db4845b83a5d9555b8717c','b59ffb92529b4d8283166e3c308b1b4c','ce96b46cd01c4fb2aa5370dcecf6ae67'}
RED={'d36f2d53448c4473ba1af7edd85312d2','6d14a961dab142c99ee19d72257c1c6e'}
VERSION=1

def finish_top(obj):
    if not obj or obj.type!='MESH':return
    identifier=obj.get('gs_id')
    if identifier in MEN:chest_contour(obj)
    if identifier in RED:red_panel(obj)

def chest_contour(obj):
    mesh=obj.data
    if mesh.get('gs_chest_contour_version',0)>=VERSION:return
    data=mesh.shape_keys.key_blocks[0].data if mesh.shape_keys else mesh.vertices
    co=np.array([v.co[:] for v in data]);delta=np.zeros_like(co)
    # The 257 ring protruded beyond both neighbouring chest rings. Remove
    # that isolated drafting corner, leaving the neck, sleeves and lower intact.
    lower=np.isclose(co[:,1],236,atol=.03);upper=np.isclose(co[:,1],280,atol=.03)
    if not lower.any() or not upper.any():raise ValueError('Unexpected kurta chest topology')
    r0=np.max(abs(co[lower,0]));r1=np.max(abs(co[upper,0]))
    for y in np.unique(co[(co[:,1]>236.03)&(co[:,1]<279.97),1]):
        ring=np.isclose(co[:,1],y,atol=.001);r=np.max(abs(co[ring,0]))
        t=(y-236)/44;target=r0+(r1-r0)*(t*t*(3-2*t))
        edge=np.clip((abs(co[ring,0])-22)/15,0,1);edge=edge*edge*(3-2*edge)
        delta[ring,0]=co[ring,0]*(target/r-1)*edge
    for i in np.flatnonzero(abs(delta[:,0])>1e-8):mesh.vertices[int(i)].co.x+=float(delta[i,0])
    if mesh.shape_keys:
        for key in mesh.shape_keys.key_blocks:
            for i in np.flatnonzero(abs(delta[:,0])>1e-8):key.data[int(i)].co.x+=float(delta[i,0])
    mesh.update();mesh['gs_chest_contour_version']=VERSION

def red_panel(obj):
    mesh=obj.data
    if mesh.get('gs_red_panel_version',0)>=VERSION:return
    uv=mesh.uv_layers.get('GS Photo UV');region=mesh.attributes.get('GS_pattern_region')
    if uv is None or region is None:raise ValueError('Red panel UV metadata missing')
    image,point,rgb=dominant_sample(obj)
    attr=mesh.attributes.get('GS_photo_shadow_repair') or mesh.attributes.new('GS_photo_shadow_repair','FLOAT','CORNER')
    # A rest-space mask lifts only the baked dark sleeve shadow. UVs and
    # embroidered highlights stay untouched; actual 3D lighting still applies.
    for poly in mesh.polygons:
        if poly.material_index!=0 or region.data[poly.index].value:continue
        for li in poly.loop_indices:
            u,v=uv.data[li].uv;down=1-v
            fade=min(1.,max(0.,(down-.035)/.04))*min(1.,max(0.,(.59-down)/.08))
            t=min(1.,max(0.,(abs(u-.5)-.125)/.055));t=t*t*(3-2*t)
            attr.data[li].value=t*fade
    material=mesh.materials[0].copy();material.name=obj.get('gs_label','Red')+' corrected photo shadow'
    mesh.materials[0]=material
    n=material.node_tree.nodes;l=material.node_tree.links
    bsdf=n.get('Principled BSDF');original=bsdf.inputs['Base Color'].links[0].from_socket
    sample=n.new('ShaderNodeTexImage');sample.image=image
    coordinate=n.new('ShaderNodeCombineXYZ');coordinate.inputs['X'].default_value=point[0];coordinate.inputs['Y'].default_value=point[1]
    l.new(coordinate.outputs[0],sample.inputs['Vector'])
    lum=n.new('ShaderNodeRGBToBW');l.new(original,lum.inputs[0])
    target=n.new('ShaderNodeRGBToBW');l.new(sample.outputs['Color'],target.inputs[0])
    ratio=n.new('ShaderNodeMath');ratio.operation='DIVIDE';l.new(lum.outputs[0],ratio.inputs[0]);l.new(target.outputs[0],ratio.inputs[1])
    ramp=n.new('ShaderNodeMapRange');ramp.clamp=True;ramp.interpolation_type='SMOOTHSTEP'
    ramp.inputs['From Min'].default_value=.25;ramp.inputs['From Max'].default_value=.95
    ramp.inputs['To Min'].default_value=1.;ramp.inputs['To Max'].default_value=0.;l.new(ratio.outputs[0],ramp.inputs['Value'])
    mask=n.new('ShaderNodeAttribute');mask.attribute_name=attr.name
    factor=n.new('ShaderNodeMath');factor.operation='MULTIPLY';l.new(mask.outputs['Fac'],factor.inputs[0]);l.new(ramp.outputs[0],factor.inputs[1])
    mix=n.new('ShaderNodeMixRGB');l.new(factor.outputs[0],mix.inputs[0]);l.new(original,mix.inputs[1]);l.new(sample.outputs['Color'],mix.inputs[2]);l.new(mix.outputs[0],bsdf.inputs['Base Color'])
    mesh['gs_red_panel_version']=VERSION

def dominant_sample(top):
    """Select a real primary-cloth texel, excluding trim and neutral background."""
    import json
    mat=top.data.materials[0]
    image=next(n.image for n in mat.node_tree.nodes if n.type=='TEX_IMAGE' and n.image)
    w,h=image.size;a=np.asarray(image.pixels[:],dtype=np.float32).reshape(h,w,4)
    profiles=json.loads(top.get('gs_photo_profiles','[]'))
    if profiles:
        p=profiles[0];left,up,right,bottom=p.get('source_region',(0,0,1,1))
        cx=p['centre'];span=p['span']
        x0=left+(cx-.28*span)*(right-left);x1=left+(cx+.28*span)*(right-left)
        y0=up+(p['top']+.28*(p['bottom']-p['top']))*(bottom-up)
        y1=up+(p['top']+.85*(p['bottom']-p['top']))*(bottom-up)
    else:x0,x1,y0,y1=.12,.88,.25,.85
    xs=np.arange(max(0,int(x0*w)),min(w,int(x1*w)),4)
    ys=np.arange(max(0,int((1-y1)*h)),min(h,int((1-y0)*h)),4)
    xx,yy=np.meshgrid(xs,ys);rgb=a[yy,xx,:3].reshape(-1,3);uv=np.c_[xx.ravel(),yy.ravel()]
    keep=(rgb.max(1)>.10)&(rgb.min(1)<.94)
    rgb=rgb[keep];uv=uv[keep]
    if not len(rgb):raise ValueError('No primary cloth samples')
    bins=np.clip((rgb*10).astype(int),0,9);codes=bins[:,0]*100+bins[:,1]*10+bins[:,2]
    dominant=np.bincount(codes,minlength=1000).argmax();chosen=rgb[codes==dominant]
    median=np.median(chosen,axis=0);i=np.argmin(((rgb-median)**2).sum(1))
    return image,((float(uv[i,0])+.5)/w,(float(uv[i,1])+.5)/h),rgb[i].tolist()

def match_dupatta(actor,top):
    if not actor or not top or top.get('gs_category')!='FEMALE_KAMEEZ':return
    identifier=str(top.get('gs_id',''))
    if not identifier:return
    if actor.data.materials and all(m and m.get('gs_matching_top')==identifier for m in actor.data.materials):
        actor['gs_matching_top']=identifier
        return
    image,point,rgb=dominant_sample(top)
    for slot,m in enumerate(list(actor.data.materials)):
        if not m:continue
        source=bpy.data.materials.get(m.get('gs_matching_source_material','')) or m
        name='AR matching Dupatta '+identifier+' '+str(slot)
        mat=bpy.data.materials.get(name)
        if mat is None:
            mat=source.copy();mat.name=name;mat['gs_matching_source_material']=source.name
            n=mat.node_tree.nodes;l=mat.node_tree.links;bsdf=n.get('Principled BSDF')
            old=bsdf.inputs['Base Color'].links[0].from_socket if bsdf.inputs['Base Color'].is_linked else None
            sample=n.new('ShaderNodeTexImage');sample.name='Kameez primary cloth';sample.image=image
            coordinate=n.new('ShaderNodeCombineXYZ');coordinate.inputs['X'].default_value=point[0];coordinate.inputs['Y'].default_value=point[1]
            l.new(coordinate.outputs[0],sample.inputs['Vector'])
            if old:
                # Desaturate the old pink textile while retaining subtle weave
                # contrast. All original alpha/roughness and drape folds survive.
                lum=n.new('ShaderNodeRGBToBW');l.new(old,lum.inputs[0])
                remap=n.new('ShaderNodeMapRange');remap.clamp=True
                remap.inputs['From Min'].default_value=.10;remap.inputs['From Max'].default_value=.65
                remap.inputs['To Min'].default_value=.88;remap.inputs['To Max'].default_value=1.12;l.new(lum.outputs[0],remap.inputs['Value'])
                multiply=n.new('ShaderNodeMixRGB');multiply.blend_type='MULTIPLY';multiply.inputs[0].default_value=1.
                l.new(sample.outputs['Color'],multiply.inputs[1]);l.new(remap.outputs[0],multiply.inputs[2]);l.new(multiply.outputs[0],bsdf.inputs['Base Color'])
            else:l.new(sample.outputs['Color'],bsdf.inputs['Base Color'])
            mat['gs_primary_srgb']=rgb;mat['gs_matching_top']=identifier
            mat.diffuse_color=(*rgb,source.diffuse_color[3])
        actor.data.materials[slot]=mat
    actor['gs_matching_top']=identifier
