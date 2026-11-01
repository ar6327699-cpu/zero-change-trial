"""Photo-informed Female variants with stable topology and permanent panel UVs.

Visible motifs are sampled from original packed photos, not regenerated. Side
depth and unseen seam fabric remain conservative priors requiring review.
"""
import json
import bpy
import numpy as np
import eastern_photos as ep

def colored_product_mask(a,category):
    # The supplied colored cloth is separated from neutral studio shadows by
    # chroma, not background brightness. White/neutral future photos retain
    # the conservative generic detector and its ambiguity checks.
    rgb=a[:,:,:3];chroma=rgb.max(axis=2)-rgb.min(axis=2)
    candidate=ep.filled_mask(chroma>.09)
    if .12<candidate.mean()<.82:return candidate
    return ep.product_mask(a,category)

def adapt(obj,profiles,category):
    co=np.array([v.co[:] for v in obj.data.vertices]);original=co.copy();top=category=='FEMALE_KAMEEZ'
    widths=np.mean([p['width_profile'] for p in profiles],axis=0)
    if top:
        target=np.clip(widths/max(widths[2],.01),.65,1.22)
        # Master shaped torso is normalized by its corresponding drafted widths.
        baseline=np.array([.82,.86,1.,1.05]);factor=np.clip(target/baseline,.94,1.12)
        down=np.clip((316-co[:,1])/220,0,1);body=np.clip((285-co[:,1])/23,0,1)
        horizontal=1+(np.interp(down,[.30,.48,.70,.92],factor)-1)*body
        # Regional radial scale, valid for subdivided rings as well as their
        # control cage. Never normalize individual vertices to a ring radius.
        below=original[:,1]<=219
        drafted=np.interp(original[:,1],[96,123,150,176,199,219],[49,48,47,46,46,42])
        minimum=np.interp(original[:,1],[96,176,199,219],[47,45,44,39])
        horizontal[below]=np.maximum(horizontal[below],minimum[below]/drafted[below])
        co[:,0]*=horizontal
        length=float(np.clip(np.mean([p['length_to_width'] for p in profiles])/2.55,.90,1.13))
        co[:,1]-=(length-1)*220*np.clip((230-co[:,1])/134,0,1)
        sleeve=float(np.clip(np.mean([p['sleeve_end'] for p in profiles])/.62,.73,1.02))
        t=np.clip((abs(original[:,0])-40)/103,0,1);sleeves=(abs(original[:,0])>39)&(original[:,1]>275)
        co[sleeves,0]=np.sign(original[sleeves,0])*(40+103*t[sleeves]*sleeve)
        cuff=float(np.clip(np.mean([p['sleeve_width']/p['span'] for p in profiles])/.25,.75,1.4))
        cy=295+5*np.minimum(1,t*3)
        co[sleeves,1]=cy[sleeves]+(co[sleeves,1]-cy[sleeves])*(1+(cuff-1)*t[sleeves])
        co[sleeves,2]=-10+(co[sleeves,2]+10)*(1+(cuff-1)*t[sleeves])
        params={'length_scale':length,'width_profile':factor.tolist(),'sleeve_scale':sleeve,'cuff_scale':cuff,
                'slit_height':199.,'neckline':'round/notch prior','depth':'independent Female master prior'}
    else:
        taper=float(np.clip((widths[-1]/max(widths[0],.01))/.52,.82,1.22))
        centre=np.where(co[:,0]>=0,18.5,-18.5);t=np.clip((130-co[:,1])/101,0,1)
        co[:,0]=centre+(co[:,0]-centre)*(1+(taper-1)*t)
        params={'ankle_taper_scale':taper,'crotch':'shared saddle; no remesh','depth':'Female waist/hip prior'}
    obj.data.vertices.foreach_set('co',co.ravel());obj.data.update()
    if top:
        # Photo sleeve length adaptation moves vertices, not skeleton joints.
        # Rebind elbow falloff to the shared rig after adaptation, otherwise a
        # three-quarter sleeve bends at the old photo-space elbow position.
        for sign,side in ((1,'Left'),(-1,'Right')):
            elbow=abs(obj.parent.data.bones['mixamorig:'+side+'ForeArm'].head_local.x)
            upper=obj.vertex_groups['mixamorig:'+side+'Arm'];fore=obj.vertex_groups['mixamorig:'+side+'ForeArm']
            for i,(x,y,z) in enumerate(co):
                if sign*x<=54 or original[i,1]<=275:continue
                t=float(np.clip((abs(x)-elbow+24)/48,0,1));t=t*t*(3-2*t)
                for g in obj.vertex_groups:g.remove([i])
                if t<1:upper.add([i],1-t,'REPLACE')
                if t>0:fore.add([i],t,'REPLACE')
    obj['gs_design_parameters']=json.dumps(params)
    return original,co,params

def material(image,label,profile):
    m=bpy.data.materials.new(label);m.use_nodes=True;n=m.node_tree.nodes;l=m.node_tree.links
    bsdf=n.get('Principled BSDF')
    for key,value in [('Metallic',0.),('Coat Weight',0.),('Roughness',.88),('Specular IOR Level',.16),('Sheen Weight',.08)]:bsdf.inputs[key].default_value=value
    uv=n.new('ShaderNodeUVMap');uv.uv_map='GS Photo UV';tex=n.new('ShaderNodeTexImage');tex.image=image;tex.extension='EXTEND'
    l.new(uv.outputs['UV'],tex.inputs['Vector'])
    # Extreme side seams are unseen in the photos. Blend into sampled cloth,
    # rather than stretching a one-pixel embroidery stripe around the torso.
    w,h=image.size;a=np.asarray(image.pixels[:],np.float32).reshape(h,w,4)[::-1]
    left,up,right,bottom=profile.get('source_region',(0,0,1,1))
    x0,x1=round(left*w),round(right*w);y0,y1=round(up*h),round(bottom*h)
    small=a[y0:y1:4,x0:x1:4,:3];chroma=small.max(2)-small.min(2);ys,xs=np.where(chroma>.10)
    if len(ys):
        values=small[ys,xs];brightness=values.mean(1);target=np.percentile(brightness,38);idx=int(np.argmin(abs(brightness-target)))
        u=(x0+int(xs[idx])*4+.5)/w;v=1-(y0+int(ys[idx])*4+.5)/h
    else:u,v=ep.source_uv(profile['centre'],1-(profile['top']+profile['bottom'])/2,profile)
    sample=n.new('ShaderNodeTexImage');sample.image=image
    point=n.new('ShaderNodeCombineXYZ');point.inputs['X'].default_value=u;point.inputs['Y'].default_value=v;l.new(point.outputs[0],sample.inputs['Vector'])
    attr=n.new('ShaderNodeVertexColor');attr.layer_name='GS_side_blend';blend=n.new('ShaderNodeMixRGB')
    l.new(attr.outputs['Color'],blend.inputs[0]);l.new(tex.outputs['Color'],blend.inputs[1]);l.new(sample.outputs['Color'],blend.inputs[2]);l.new(blend.outputs[0],bsdf.inputs['Base Color'])
    # No generated paint, glow, camera exposure multiplier or aggressive bump.
    return m

def apply(obj,front,back,category,flip_back=True,regions=None):
    top=category=='FEMALE_KAMEEZ';kind='KURTA' if top else 'PAJAMA'
    records=[ep.read(path,obj.name+' '+label,kind,regions[i] if regions else None,mask_function=colored_product_mask)
             for i,(label,path) in enumerate([('Front',front),('Back',back)])]
    profiles=[r[1] for r in records]
    if max(p['length_to_width'] for p in profiles)/min(p['length_to_width'] for p in profiles)>1.35:raise ValueError('Female front/back proportions disagree: '+json.dumps(profiles))
    original,co,params=adapt(obj,profiles,category)
    uv=obj.data.uv_layers.get('GS Photo UV') or obj.data.uv_layers.new(name='GS Photo UV');obj.data.uv_layers.active=uv
    obj.data.materials.clear()
    for image,p,mask in records:obj.data.materials.append(material(image,image.name+' matte cotton',p))
    edge=obj.data.color_attributes.get('GS_side_blend') or obj.data.color_attributes.new(name='GS_side_blend',type='FLOAT_COLOR',domain='CORNER')
    for loop in obj.data.loops:
        normal=abs(obj.data.vertices[loop.vertex_index].normal.z)
        t=float(np.clip((.50-normal)/.30,0,1));t=t*t*(3-2*t);edge.data[loop.index].color=(t,t,t,1)
    regions_attr=obj.data.attributes['GS_pattern_region'];high=316 if top else 213;low=96 if top else 29
    # Face choice is fixed in REST space. No camera-facing material switching.
    for poly in obj.data.polygons:
        side=0 if poly.normal.z>=0 else 1;poly.material_index=side
        _,p,mask=records[side];mh,mw=mask.shape;region=regions_attr.data[poly.index].value
        for li in poly.loop_indices:
            i=obj.data.loops[li].vertex_index;x,y,z=original[i];ix=-x if side and flip_back else x
            d=(high-y)/(high-low);u=p['centre']+ix/(98 if top else 92)*p['span']
            v=p['top']+d*(p['bottom']-p['top'])
            if top and region:
                t=float(np.clip((abs(x)-40)/103,0,1));around=float(np.clip((y-(295+5*min(1,t*3)))/17,-1,1));sgn=1 if ix>=0 else -1
                u=(p['centre']+sgn*p['span']*.47)*(1-t)+p['sleeve_centres'][1 if sgn>0 else 0]*t+sgn*around*p['sleeve_width']*.42
                v=p['top']+(.10+(p['sleeve_end']-.10)*t-around*.012)*(p['bottom']-p['top'])
            # Preserve central artwork, extend boundary samples only onto the
            # unknown side seam. No full scan-line stretch of floral details.
            row=int(np.clip(round(v*(mh-1)),0,mh-1));runs=ep.runs(mask[row])
            if runs:
                if top and region:
                    # Track the actual photographed sleeve strip on every row;
                    # never interpolate a cuff UV across the background gap.
                    if len(runs)>=3:strip=runs[-1] if sgn>0 else runs[0]
                    else:
                        full=max(runs,key=len);middle=p['centre']*(mw-1)
                        strip=full[full>middle+p['span']*mw*.38] if sgn>0 else full[full<middle-p['span']*mw*.38]
                        if len(strip)<4:strip=full
                    inset=min(2.5,(strip[-1]-strip[0])*.18)
                    lo=(strip[0]+inset)/(mw-1);hi=(strip[-1]-inset)/(mw-1)
                    u=(lo+hi)*.5+sgn*around*(hi-lo)*.5
                else:
                    centre=p['centre']*(mw-1)
                    strips=[r for r in runs if r[0]<=centre<=r[-1]] if top else []
                    strip=strips[0] if strips else min(runs,key=lambda r:abs((r[0]+r[-1])*.5-u*(mw-1)))
                    inset=min(2.5,(strip[-1]-strip[0])*.1)
                    u=float(np.clip(u,(strip[0]+inset)/(mw-1),(strip[-1]-inset)/(mw-1)))
            uv.data[li].uv=ep.source_uv(float(np.clip(u,0,1)),float(np.clip(1-v,0,1)),p)
    obj['gs_photo_method']='Female master v1: fixed front/back/sleeve UV panels; source motifs preserved'
    obj['gs_photo_profiles']=json.dumps(profiles)
    return ['Approximate photo-informed silhouette. Unseen depth/seams are priors; live fit requires review.']
