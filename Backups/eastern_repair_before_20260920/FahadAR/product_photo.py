"""Product-photo alignment: stable torso mapping; no scan-line logo warping."""
from collections import deque
import numpy as np
import bpy

def silhouette(a):
    h,w=a.shape[:2]
    border=np.concatenate([a[:6,:,:3].reshape(-1,3),a[-6:,:,:3].reshape(-1,3),a[:,:6,:3].reshape(-1,3),a[:,-6:,:3].reshape(-1,3)])
    bg=np.median(border,axis=0);noise=np.median(np.linalg.norm(border-bg,axis=1))
    threshold=np.clip(4*noise,.022,.07)
    left=np.median(a[:,:4,:3],axis=1);right=np.median(a[:,-4:,:3],axis=1)
    choose_left=np.linalg.norm(left-bg,axis=1)<=np.linalg.norm(right-bg,axis=1)
    row_bg=np.where(choose_left[:,None],left,right)
    mask=np.linalg.norm(a[:,:,:3]-row_bg[:,None,:],axis=2)>threshold
    if np.any(a[:,:,3]<.1):mask=a[:,:,3]>.5
    # Keep the largest connected garment, not floating shadows or a badge.
    seen=np.zeros((h,w),bool);best=[]
    for y,x in zip(*np.where(mask)):
        if seen[y,x]:continue
        q=deque([(int(y),int(x))]);seen[y,x]=True;component=[]
        while q:
            yy,xx=q.popleft();component.append((yy,xx))
            for ny,nx in [(yy-1,xx),(yy+1,xx),(yy,xx-1),(yy,xx+1)]:
                if 0<=ny<h and 0<=nx<w and mask[ny,nx] and not seen[ny,nx]:seen[ny,nx]=True;q.append((ny,nx))
        if len(component)>len(best):best=component
    if len(best)<h*w*.08:raise ValueError('Cannot safely locate the whole garment. Use a clearer background.')
    out=np.zeros((h,w),bool);ys,xs=zip(*best);out[ys,xs]=True
    return out

def analyse(a):
    # Input uses top-down coordinates; measurements remain normalized.
    h,w=a.shape[:2];step=max(1,int(np.ceil(max(h,w)/384)))
    mask=silhouette(a[::step,::step]);yy,xx=np.where(mask)
    top,bottom=yy.min(),yy.max();height=bottom-top
    # Lower torso is unaffected by the downward-facing product-photo sleeves.
    band=mask[int(top+height*.48):int(top+height*.82)]
    left=[];right=[]
    for row in band:
        ids=np.flatnonzero(row)
        if len(ids)>8:left.append(ids[0]);right.append(ids[-1])
    if len(left)<5:raise ValueError('Garment torso is unclear')
    left=float(np.median(left));right=float(np.median(right));centre=(left+right)*.5
    width=right-left
    if width<12:raise ValueError('Garment is too small')
    # All torso features share one affine horizontal mapping. Do not recenter
    # each row, which bends lettering and moves chest graphics sideways.
    result={'left':left*step/w,'right':right*step/w,'centre':centre*step/w,
            'top':float(top*step/h),'bottom':float(bottom*step/h),'width':width*step/w,
            'confidence':'approximate silhouette; inspect preview'}
    outline=[]
    for column in mask.T:
        ids=np.flatnonzero(column)
        outline.append(float(ids.min()*step/h) if len(ids) else float('nan'))
    outline=np.array(outline);valid=np.flatnonzero(np.isfinite(outline))
    outline=np.interp(np.arange(len(outline)),valid,outline[valid])
    result['top_outline']=outline.tolist()
    # Front neckline shape: central opening depth compared to its width.
    edges=[]
    for x in range(max(0,int(centre-width*.20)),min(mask.shape[1],int(centre+width*.20)+1)):
        ids=np.flatnonzero(mask[:,x])
        if len(ids):edges.append((x,float(ids.min())))
    if edges:
        e=np.array(edges);central=e[abs(e[:,0]-centre)<width*.05,1]
        outside=e[abs(e[:,0]-centre)>width*.13,1]
        depth=(float(np.median(central))-float(np.median(outside))) if len(central) and len(outside) else 0
        result['neckline']='V' if depth>width*.16 else 'CREW'
        result['neck_depth_ratio']=depth/max(width,1)
    else:result['neckline']='CREW'
    return result,mask

def tailor(obj,neckline='CREW',slim=.08):
    """Keep shoulder pivots/rest rig; change only this new garment's geometry."""
    c=np.array([v.co[:] for v in obj.data.vertices]);x,y,z=c.T.copy()
    if neckline=='CREW':
        central=np.clip(1-(abs(x)/19)**2,0,1)
        front=np.clip((z+4)/12,0,1)
        upper=np.clip((y-274)/23,0,1);upper=upper*upper*(3-2*upper)
        c[:,1]+=15*central*front*upper
    # Shoulder x=+/-39 and sleeves stay fixed. Blend the narrower torso out
    # below the shoulders; do not shrink the skeleton/camera calibration.
    region=np.clip((290-y)/35,0,1)*np.clip((42-abs(x))/12,0,1)
    c[:,0]*=1-float(slim)*region
    c[:,2]=-5+(c[:,2]+5)*(1-float(slim)*1.4*region)
    obj.data.vertices.foreach_set('co',c.ravel());obj.data.update()
    if obj.data.has_custom_normals:
        obj.data.normals_split_custom_set([(0.,0.,0.)]*len(obj.data.loops))
    obj['gs_neckline']=neckline;obj['gs_slim']=float(slim)

def apply(obj,front_path,back_path,neckline='AUTO',slim=.08,flip_back=True):
    from pathlib import Path
    records=[];images=[]
    for side,path in [('Front',front_path),('Back',back_path)]:
        p=Path(bpy.path.abspath(path))
        if not p.is_file():raise ValueError('Front/back photo is missing')
        im=bpy.data.images.load(str(p),check_existing=False)
        w,h=im.size
        if w*h>40000000 or min(w,h)<64:raise ValueError('Photo size unsupported')
        if max(w,h)>1024:
            k=1024/max(w,h);im.scale(round(w*k),round(h*k));w,h=im.size
        a=np.array(im.pixels[:],np.float32).reshape(h,w,4)[::-1].copy()
        profile,mask=analyse(a);records.append(profile)
        # Reuse the original encoded photo. Do NOT create a linear-colour
        # replacement image from its sRGB pixels (that washes black to grey).
        im.name=obj.name+' '+side+' source';im.pack();images.append(im)
    chosen=records[0]['neckline'] if neckline=='AUTO' else neckline
    tailor(obj,chosen,slim)
    c=np.array([v.co[:] for v in obj.data.vertices]);low=float(c[:,1].min());high=float(c[:,1].max())
    # Torso half-width below the armpits, not sleeve-to-sleeve width.
    band=c[(c[:,1]>235)&(c[:,1]<270)]
    half=float(np.percentile(abs(band[:,0]),98))
    uv=obj.data.uv_layers.get('GS Photo UV') or obj.data.uv_layers.new(name='GS Photo UV')
    obj.data.uv_layers.active=uv;obj.data.materials.clear()
    for im in images:
        mat=bpy.data.materials.new(im.name+' fabric');mat.use_nodes=True
        bsdf=mat.node_tree.nodes.get('Principled BSDF');bsdf.inputs['Roughness'].default_value=.82
        bsdf.inputs['Specular IOR Level'].default_value=.18
        tex=mat.node_tree.nodes.new('ShaderNodeTexImage');tex.image=im;tex.extension='EXTEND'
        u=mat.node_tree.nodes.new('ShaderNodeUVMap');u.uv_map='GS Photo UV'
        mat.node_tree.links.new(u.outputs['UV'],tex.inputs['Vector']);mat.node_tree.links.new(tex.outputs['Color'],bsdf.inputs['Base Color'])
        obj.data.materials.append(mat)
    for poly in obj.data.polygons:
        side=0 if poly.normal.z>=0 else 1;poly.material_index=side;r=records[side]
        for li in poly.loop_indices:
            x,y,z=c[obj.data.loops[li].vertex_index]
            bodyx=x/half
            if side and flip_back:bodyx=-bodyx
            # Stable torso affine coordinates preserve logo aspect/centering.
            u=r['centre']+bodyx*(r['right']-r['left'])*.5
            down=(high-y)/max(high-low,1)
            v=r['top']+down*(r['bottom']-r['top'])
            # Product sleeves point down in photographs, while the rig's rest
            # sleeves extend sideways. Rotate this region in texture space,
            # keeping the central chest/logo independent of that mapping.
            blend=float(np.clip((abs(x)-30)/15,0,1));blend=blend*blend*(3-2*blend)
            if blend:
                sign=1 if bodyx>=0 else -1
                along=max(0,abs(x)-38)/max(abs(c[:,0]).max()-38,1)
                around=(y-299)/20
                su=r['centre']+sign*(r['right']-r['left'])*(.47+along*.20+around*.17)
                sv=r['top']+(r['bottom']-r['top'])*(.09+along*.28-around*.08)
                u=u*(1-blend)+su*blend;v=v*(1-blend)+sv*blend
            outline=r['top_outline'];edge=outline[min(max(round(u*(len(outline)-1)),0),len(outline)-1)]
            upper=float(np.clip(1-(v-r['top'])/max((r['bottom']-r['top'])*.32,.01),0,1))
            v+=max(0,edge-r['top'])*upper*upper
            v=max(v,edge+.003)
            uv.data[li].uv=(float(np.clip(u,0,1)),float(np.clip(1-v,0,1)))
    obj.data.update();obj['gs_photo_method']='anchored product photo v2'
    import json
    obj['gs_photo_profiles']=json.dumps(records)
    return ['Product-photo alignment is approximate; inspect neckline, print and sleeve seams.']
