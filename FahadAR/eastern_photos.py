"""Eastern-only product analysis and stable panel UVs on master copies.

No wearer measurements and no claim of cloth reconstruction. Unsupported or
ambiguous backgrounds fail before saving rather than silently changing fit.
"""
import json
from collections import deque
from pathlib import Path
import bpy
import numpy as np
from product_photo import silhouette

def _clean_texture_background(image, rgba, mask, crop_box=None):
    """Replace photo backdrop outside the detected cloth with cloth color.

    Product photos are projection references, not transparent cutouts. Any UV
    that lands a few pixels outside the silhouette must stay fabric coloured;
    otherwise live deformation turns the studio backdrop into white wedges at
    hems, cuffs, and side seams.
    """
    h,w=rgba.shape[:2]
    small=np.asarray(mask,dtype=np.uint8)
    full=small[np.minimum(np.arange(h)*small.shape[0]//h,small.shape[0]-1)[:,None],np.minimum(np.arange(w)*small.shape[1]//w,small.shape[1]-1)[None,:]]
    # Erode the low-resolution outline to discard backdrop antialiasing.
    ph,pw=small.shape;padded=np.pad(small.astype(bool),1)
    safe=np.logical_and.reduce([padded[dy:dy+ph,dx:dx+pw] for dy in range(3) for dx in range(3)])
    inside=safe[np.minimum(np.arange(h)*ph//h,ph-1)[:,None],np.minimum(np.arange(w)*pw//w,pw-1)[None,:]]
    # Refine against full-resolution RGB: upsampling a 384px outline alone
    # can place a supposedly safe sample back on a bright sloping edge.
    rgb=rgba[:,:,:3];border=np.concatenate([rgb[:3].reshape(-1,3),rgb[-3:].reshape(-1,3),rgb[:,:3].reshape(-1,3),rgb[:,-3:].reshape(-1,3)])
    bg=np.median(border,0);noise=np.median(np.linalg.norm(border-bg,axis=1))
    contrast=np.linalg.norm(rgb-bg,axis=2)>float(np.clip(noise*4,.08,.2))
    cloth=rgb[inside]
    if len(cloth) and np.median(cloth.mean(1))<.25 and np.median(np.ptp(cloth,axis=1))<.04 and bg.mean()>.5:
        contrast &= rgb.mean(2)<.4
    inside &= contrast
    padded=np.pad(inside,3)
    inside=np.logical_and.reduce([padded[dy:dy+h,dx:dx+w] for dy,dx in ((0,3),(6,3),(3,0),(3,6),(3,3))])
    rows=np.flatnonzero(inside.any(1))
    if not len(rows):return
    source=rgba.copy();xx=np.arange(w)
    for y in range(h):
        sy=y if inside[y].any() else int(rows[np.argmin(abs(rows-y))])
        ids=np.flatnonzero(inside[sy]);right=np.searchsorted(ids,xx).clip(0,len(ids)-1);left=(right-1).clip(0,len(ids)-1)
        nearest=np.where(abs(ids[left]-xx)<abs(ids[right]-xx),ids[left],ids[right])
        outside=~inside[y];rgba[y,outside,:3]=source[sy,nearest[outside],:3]

def filled_mask(mask):
    # Close tiny antialiasing gaps, then fill enclosed white fabric interiors.
    # Never flood across open sleeve/leg gaps connected to the background.
    h,w=mask.shape;p=np.pad(mask,1)
    dilated=np.logical_or.reduce([p[y:y+h,x:x+w] for y in range(3) for x in range(3)])
    p=np.pad(dilated,1)
    mask=np.logical_and.reduce([p[y:y+h,x:x+w] for y in range(3) for x in range(3)])
    seen=np.zeros_like(mask);q=deque()
    for y,x in [(0,x) for x in range(w)]+[(h-1,x) for x in range(w)]+[(y,0) for y in range(h)]+[(y,w-1) for y in range(h)]:
        if not mask[y,x] and not seen[y,x]:seen[y,x]=True;q.append((y,x))
    while q:
        y,x=q.popleft()
        for yy,xx in [(y-1,x),(y+1,x),(y,x-1),(y,x+1)]:
            if 0<=yy<h and 0<=xx<w and not mask[yy,xx] and not seen[yy,xx]:seen[yy,xx]=True;q.append((yy,xx))
    return ~seen

def runs(row):
    ids=np.flatnonzero(row)
    if not len(ids):return []
    return [x for x in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1) if len(x)>2]

def product_mask(a,category):
    mask=filled_mask(silhouette(a))
    if category!='KURTA':return mask
    def centre_complete(m):
        ys,xs=np.where(m)
        return len(ys) and all(m[int(ys.min()+d*(ys.max()-ys.min())),m.shape[1]//2] for d in [.30,.48,.70,.92])
    if centre_complete(mask):return mask
    # Low-contrast white fabric fallback. Estimate background independently
    # on each scanline, rather than increasing the crop tolerance globally.
    left=np.median(a[:,:4,:3],axis=1);right=np.median(a[:,-4:,:3],axis=1)
    bg=(left+right)*.5
    noise=np.median(np.linalg.norm(left-right,axis=1))
    threshold=float(np.clip(noise*3,.006,.018))
    candidate=filled_mask(np.linalg.norm(a[:,:,:3]-bg[:,None,:],axis=2)>threshold)
    if .1<candidate.mean()<.85 and centre_complete(candidate):return candidate
    return mask

def read(path,name,category,region=None,mask_function=None):
    path=Path(bpy.path.abspath(path))
    if not path.is_file():raise ValueError('Choose an existing front/back garment photo')
    image=bpy.data.images.load(str(path),check_existing=False)
    try:
        w,h=image.size
        if min(w,h)<64 or w*h>40000000:raise ValueError('Unsupported Eastern photo resolution')
        if max(w,h)>1536:
            scale=1536/max(w,h);image.scale(round(w*scale),round(h*scale));w,h=image.size
        raw=np.asarray(image.pixels[:],dtype=np.float32).reshape(h,w,4)
        a=raw[::-1]
        # Analyze a sheet cell without editing any source image pixels. UVs
        # later address the same region in the original packed image.
        source_region=(0.,0.,1.,1.)
        if region is not None:
            l,t,r,b=region
            if not (0<=l<r<=1 and 0<=t<b<=1):raise ValueError('Invalid source photo region')
            x0,x1=round(l*w),round(r*w);y0,y1=round(t*h),round(b*h)
            source_region=(x0/w,y0/h,x1/w,y1/h)
            a=a[y0:y1,x0:x1];h,w=a.shape[:2]
        step=max(1,int(np.ceil(max(w,h)/384)));mask=(mask_function or product_mask)(a[::step,::step],category)
        _clean_texture_background(image,a,mask)
        # Write the cleaned crop back into the packed image. This keeps all UVs
        # inside the photographed garment even when the source is a two-view
        # sheet with a bright studio backdrop.
        if region is not None:
            raw[::-1][y0:y1,x0:x1]=a
        else:raw[::-1]=a
        image.pixels.foreach_set(raw.ravel());image.update()
        mh,mw=mask.shape;yy,xx=np.where(mask)
        top=float(yy.min());bottom=float(yy.max());height=bottom-top
        if height<mh*.5 or not .10<mask.mean()<.85:
            raise ValueError('Eastern garment outline uncertain: use a plain contrasting background and a complete product photo')
        bounds=[];centres=[]
        for d in [.30,.48,.70,.92]:
            samples=[]
            for y in range(int(top+height*(d-.015)),int(top+height*(d+.015))+1):
                rr=runs(mask[y])
                if not rr:continue
                if category=='KURTA':
                    mid=mw*.5
                    rr=[r for r in rr if r[0]<=mid<=r[-1]]
                    if not rr:continue
                    ids=rr[0]
                else:ids=np.concatenate(rr)
                samples.append((ids[0],ids[-1]))
            if not samples:raise ValueError(name+': cannot identify torso/leg outline at '+str(d)+'; use a contrasting plain background')
            lo,hi=np.median(samples,axis=0);bounds.append(float((hi-lo)/mw));centres.append(float((lo+hi)*.5/mw))
        centre=float(np.median(centres))
        span=float(np.median(bounds[1:3])) if category=='KURTA' else float((xx.max()-xx.min())/mw)
        sleeve_end=.61;sleeve_centres=[centre-span*.65,centre+span*.65];sleeve_width=span*.22
        if category=='KURTA':
            found=[]
            for y in range(int(top+height*.3),int(top+height*.76)):
                rr=runs(mask[y])
                if len(rr)==3 and rr[1][0]<=centre*mw<=rr[1][-1]:found.append((y,rr))
            if len(found)>6:
                y,rr=found[max(0,len(found)-4)]
                sleeve_end=float((y-top)/height)
                sleeve_centres=[float((r[0]+r[-1])*.5/mw) for r in (rr[0],rr[-1])]
                sleeve_width=float(np.mean([len(rr[0]),len(rr[-1])])/mw)
        profile={'top':top/mh,'bottom':bottom/mh,'centre':centre,'span':span,
                 'width_profile':bounds,'length_to_width':height/mh*h/(span*w),
                 'sleeve_end':sleeve_end,'sleeve_centres':sleeve_centres,
                 'sleeve_width':sleeve_width,'depth_source':'master prior',
                 'slit_source':'master prior; silhouette alone is ambiguous',
                 'confidence':'approximate; inspect before activation','source_region':source_region}
        image.name=name;image.pack()
        return image,profile,mask
    except Exception:
        bpy.data.images.remove(image)
        raise

def adapt(obj,profiles,category):
    c=np.array([v.co[:] for v in obj.data.vertices]);original=c.copy()
    if category=='KURTA':
        # Width profile is relative to the same garment's lower torso, not
        # camera pixels. Keep shoulder pivots and the person-fit solver fixed.
        widths=np.mean([p['width_profile'] for p in profiles],axis=0)
        ratios=np.clip(widths/max(widths[2],.01),.90,1.10)
        down=np.clip((324-c[:,1])/209,0,1)
        scales=np.interp(down,[.30,.48,.70,.92],ratios)
        body=np.clip((280-c[:,1])/30,0,1)
        c[:,0]*=1+(scales-1)*body
        # Safe length range: upper knee region, never stretch the armature.
        length_ratio=np.mean([p['length_to_width'] for p in profiles])
        length_scale=float(np.clip(length_ratio/2.5,.94,1.06))
        lower=np.clip((236-c[:,1])/121,0,1)
        c[:,1]-=(length_scale-1)*209*lower
        # Product length is a style ratio, not authority to extend below knees.
        # The live knee constraint additionally handles wearer proportions.
        knee=bpy.data.objects['Armature'].data.bones['mixamorig:LeftLeg'].head_local.y
        before_hem=float(c[:,1].min());target_hem=max(before_hem,float(knee)+4.)
        c[:,1]+=(target_hem-before_hem)*np.clip((236-c[:,1])/max(236-before_hem,1),0,1)
        sleeve_length=float(np.clip(np.mean([(p['sleeve_end']-.105)/.52 for p in profiles]),.94,1.05))
        cuff_scale=float(np.clip(np.mean([p['sleeve_width']/max(p['span'],.01) for p in profiles])/.22,.85,1.25))
        along=np.clip((np.abs(original[:,0])-48)/95,0,1)
        sleeve=original[:,1]>277
        c[sleeve,0]+=np.sign(original[sleeve,0])*along[sleeve]*95*(sleeve_length-1)
        scale=1+(cuff_scale-1)*along
        centre_y=295+5*np.minimum(1,along*3)
        c[sleeve,1]=centre_y[sleeve]+(c[sleeve,1]-centre_y[sleeve])*scale[sleeve]
        c[sleeve,2]=-10+(c[sleeve,2]+10)*scale[sleeve]
        params={'width_ratios':ratios.tolist(),'length_scale':length_scale,'hem_rest_height':target_hem,'length_policy':'rest above knee; live observed knee cap',
                'shoulder_policy':'fixed rig pivot','sleeve_length_scale':sleeve_length,'cuff_width_scale':cuff_scale,
                'slit_policy':'master below-hip prior','depth_policy':'unchanged master depth'}
    else:
        widths=np.mean([p['width_profile'] for p in profiles],axis=0)
        ratios=np.clip(widths/max(widths[0],.01),.25,1.05)
        # Adapt lower taper locally around each leg centre. Waist and crotch
        # remain connected; no per-image remeshing or global scale.
        target=float(np.clip(ratios[-1]/.54,.88,1.18))
        t=np.clip((125-c[:,1])/96,0,1)
        centres=np.where(c[:,0]>=0,18.5,-18.5)
        c[:,0]=centres+(c[:,0]-centres)*(1+(target-1)*t)
        params={'width_ratios':ratios.tolist(),'ankle_taper_scale':target,
                'crotch_policy':'shared master saddle','depth_policy':'unchanged master depth'}
    assert np.isfinite(c).all() and len(c)==len(original)
    obj.data.vertices.foreach_set('co',c.ravel());obj.data.update()
    obj['gs_design_parameters']=json.dumps(params)

def source_uv(u,v,profile):
    l,t,r,b=profile.get('source_region',(0,0,1,1))
    return l+u*(r-l),1-(t+(1-v)*(b-t))

def apply(obj,front,back,category,flip_back=True,regions=None):
    records=[]
    try:
        for k,(label,path) in enumerate([('Front',front),('Back',back)]):records.append(read(path,obj.name+' '+label,category,regions[k] if regions else None))
        profiles=[r[1] for r in records]
        if max(p['length_to_width'] for p in profiles)/min(p['length_to_width'] for p in profiles)>1.35:
            raise ValueError('Front/back garment proportions disagree. Use matching views of the same item.')
        adapt(obj,profiles,category)
        coords=np.array([v.co[:] for v in obj.data.vertices]);low=float(coords[:,1].min());high=float(coords[:,1].max())
        uv=obj.data.uv_layers.get('GS Photo UV') or obj.data.uv_layers.new(name='GS Photo UV')
        obj.data.uv_layers.active=uv;obj.data.materials.clear()
        # A fixed rest-space corner attribute blends extreme side seams into
        # fabric sampled from both photos. It never depends on camera angle
        # or animated normals, so turning the wearer cannot switch textures.
        attr=obj.data.color_attributes.get('GS_side_blend') or obj.data.color_attributes.new(name='GS_side_blend',type='FLOAT_COLOR',domain='CORNER')
        for loop in obj.data.loops:
            nz=abs(obj.data.vertices[loop.vertex_index].normal.z)
            # Keep fabric detail on broad front/back panels. Only the actual
            # side transition uses the unknown-side approximation.
            fade=float(np.clip((.55-nz)/.40,0,1));fade=fade*fade*(3-2*fade)
            attr.data[loop.index].color=(fade,fade,fade,1)
        for image,profile,mask in records:
            material=bpy.data.materials.new(image.name+' fabric');material.use_nodes=True
            bsdf=material.node_tree.nodes.get('Principled BSDF');bsdf.inputs['Roughness'].default_value=.85
            bsdf.inputs['Specular IOR Level'].default_value=.18
            tex=material.node_tree.nodes.new('ShaderNodeTexImage');tex.image=image;tex.extension='EXTEND'
            node=material.node_tree.nodes.new('ShaderNodeUVMap');node.uv_map='GS Photo UV'
            material.node_tree.links.new(node.outputs['UV'],tex.inputs['Vector'])
            samples=[]
            for side_image,side_profile,_ in records:
                sample=material.node_tree.nodes.new('ShaderNodeTexImage');sample.image=side_image
                point=material.node_tree.nodes.new('ShaderNodeCombineXYZ')
                u,v=source_uv(side_profile['centre']+side_profile['span']*.22,1-(side_profile['top']+(side_profile['bottom']-side_profile['top'])*.72),side_profile)
                point.inputs['X'].default_value=u
                point.inputs['Y'].default_value=v
                material.node_tree.links.new(point.outputs[0],sample.inputs['Vector'])
                samples.append(sample)
            average=material.node_tree.nodes.new('ShaderNodeMixRGB');average.inputs[0].default_value=.5
            material.node_tree.links.new(samples[0].outputs['Color'],average.inputs[1]);material.node_tree.links.new(samples[1].outputs['Color'],average.inputs[2])
            edge=material.node_tree.nodes.new('ShaderNodeVertexColor');edge.layer_name='GS_side_blend'
            blend=material.node_tree.nodes.new('ShaderNodeMixRGB')
            material.node_tree.links.new(edge.outputs['Color'],blend.inputs[0]);material.node_tree.links.new(tex.outputs['Color'],blend.inputs[1])
            material.node_tree.links.new(average.outputs['Color'],blend.inputs[2]);material.node_tree.links.new(blend.outputs['Color'],bsdf.inputs['Base Color'])
            obj.data.materials.append(material)
        regions=obj.data.attributes.get('GS_pattern_region')
        for poly in obj.data.polygons:
            side=0 if poly.normal.z>=0 else 1;poly.material_index=side
            _,p,mask=records[side];mh,mw=mask.shape
            region=regions.data[poly.index].value if regions else 0
            for li in poly.loop_indices:
                x,y,z=coords[obj.data.loops[li].vertex_index];image_x=-x if side and flip_back else x
                down=(high-y)/max(high-low,1)
                half=49 if category=='KURTA' else 53
                u=p['centre']+image_x/(2*half)*p['span']
                v=p['top']+down*(p['bottom']-p['top'])
                if category=='KURTA' and region:
                    t=float(np.clip((abs(x)-42)/101,0,1));around=float(np.clip((y-(295+5*min(1,t*3)))/17,-1,1))
                    s=1 if image_x>=0 else -1
                    start=p['centre']+s*p['span']*.46
                    end=p['sleeve_centres'][1 if s>0 else 0]
                    u=start*(1-t)+end*t+s*around*p['sleeve_width']*.42
                    v=p['top']+(.105+(p['sleeve_end']-.105)*t-around*.018)*(p['bottom']-p['top'])
                # Edge extension in UV space only: central print/placket stays
                # affine, no scan-line squeezing of the whole torso.
                iy=int(np.clip(round(v*(mh-1)),0,mh-1));ids=np.flatnonzero(mask[iy])
                if len(ids):
                    ix=int(np.clip(round(u*(mw-1)),0,mw-1))
                    if not mask[iy,ix]:u=float(ids[np.argmin(abs(ids-ix))]/(mw-1))
                uv.data[li].uv=source_uv(float(np.clip(u,0,1)),float(np.clip(1-v,0,1)),p)
        obj['gs_photo_method']='Eastern master v2: normalized product profiles + anchored panel UV'
        obj['gs_photo_profiles']=json.dumps(profiles)
        obj.data.update()
        return ['Approximate Eastern pattern; inspect seams, silhouette and live fit. Depth/slits use master priors.']
    except Exception:
        for image,_,_ in records:
            if image.users==0:bpy.data.images.remove(image)
        raise
