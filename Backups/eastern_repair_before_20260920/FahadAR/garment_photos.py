"""Deterministic, offline photo preparation and two-sided template projection.

No generative AI, body-size inference, cloud upload, or paid model. The existing
MediaPipe AI continues to handle live tracking. Plain-background flat-lays are
required; uncertain background separation fails instead of silently claiming fit.
"""
from pathlib import Path
from collections import deque
import bpy,numpy as np

def prepare_photo(path,name,auto_crop=True,threshold=.16):
    path=Path(bpy.path.abspath(path)).resolve()
    if not path.is_file() or path.suffix.lower() not in {'.png','.jpg','.jpeg','.webp','.tif','.tiff'}:
        raise ValueError('Choose an existing PNG/JPG/WebP/TIFF garment photo')
    raw=bpy.data.images.load(str(path),check_existing=False)
    try:
        w,h=raw.size
        if min(w,h)<64 or w*h>40000000:raise ValueError('Photo must be at least 64px and at most 40 megapixels')
        if max(w,h)>1024:
            scale=1024/max(w,h);raw.scale(round(w*scale),round(h*scale));w,h=raw.size
        pixels=np.empty(w*h*4,np.float32);raw.pixels.foreach_get(pixels);pixels=pixels.reshape(h,w,4)
        mask=np.ones((h,w),bool);notes=[]
        if auto_crop:
            # Work on a <=256px mask; flood only border-connected background.
            step=max(1,int(np.ceil(max(w,h)/256)));small=pixels[::step,::step]
            if np.any(small[:,:,3]<.1):mask_small=small[:,:,3]>.5
            else:
                border=np.concatenate([small[0,:,:3],small[-1,:,:3],small[:,0,:3],small[:,-1,:3]])
                colour=np.median(border,axis=0)
                if np.median(np.linalg.norm(border-colour,axis=1))>.18:
                    raise ValueError('Background is too varied. Use a plain contrasting background or a transparent PNG; alternatively crop the photo yourself and disable Auto crop.')
                possible=np.linalg.norm(small[:,:,:3]-colour,axis=2)<threshold
                sh,sw=possible.shape;visited=np.zeros_like(possible);q=deque()
                for y,x in [(0,x) for x in range(sw)]+[(sh-1,x) for x in range(sw)]+[(y,0) for y in range(sh)]+[(y,sw-1) for y in range(sh)]:
                    if possible[y,x] and not visited[y,x]:visited[y,x]=True;q.append((y,x))
                while q:
                    y,x=q.popleft()
                    for yy,xx in [(y-1,x),(y+1,x),(y,x-1),(y,x+1)]:
                        if 0<=yy<sh and 0<=xx<sw and possible[yy,xx] and not visited[yy,xx]:visited[yy,xx]=True;q.append((yy,xx))
                mask_small=~visited
            fraction=mask_small.mean()
            if not .04<fraction<.94:raise ValueError('Garment/background separation is uncertain. Use stronger background contrast or disable Auto crop after manually cropping.')
            # Conservative one-cell inset excludes mixed background pixels at
            # anti-aliased garment edges before extending cloth colours.
            padded=np.pad(mask_small,1,constant_values=False)
            sh,sw=mask_small.shape
            mask_small=np.logical_and.reduce([padded[dy:dy+sh,dx:dx+sw] for dy in range(3) for dx in range(3)])
            if not mask_small.any():raise ValueError('Garment silhouette is too small/thin for automatic cropping')
            mask=np.repeat(np.repeat(mask_small,step,axis=0),step,axis=1)[:h,:w]
            yy,xx=np.where(mask);x0,x1=int(xx.min()),int(xx.max()+1);y0,y1=int(yy.min()),int(yy.max()+1)
            if x0==0 or y0==0 or x1==w or y1==h:notes.append('Garment touches photo edge: check for cropped sleeves/hem.')
            pixels=pixels[y0:y1,x0:x1].copy();mask=mask[y0:y1,x0:x1]
        h,w=mask.shape
        # Extend garment colours into background to prevent white borders on
        # template seams. The mask never cuts holes into the garment geometry.
        valid_rows=np.flatnonzero(mask.any(axis=1))
        if not len(valid_rows):raise ValueError('No garment pixels found')
        bounds=np.zeros((h,2),np.float32)
        for y in range(h):
            ids=np.flatnonzero(mask[y])
            if not len(ids):
                src=valid_rows[np.argmin(abs(valid_rows-y))];pixels[y]=pixels[src];ids=np.flatnonzero(mask[src])
            bounds[y]=[ids[0]/max(w-1,1),ids[-1]/max(w-1,1)]
            for c in range(3):pixels[y,:,c]=np.interp(np.arange(w),ids,pixels[y,ids,c])
        pixels[:,:,3]=1
        image=bpy.data.images.new(name,width=w,height=h,alpha=True)
        image.colorspace_settings.name='sRGB'
        image.pixels.foreach_set(pixels.ravel());image.update();image.pack()
        return image,bounds,notes
    finally:
        bpy.data.images.remove(raw)

def apply_photos(obj,front,back,auto_crop=True,threshold=.16,flip_back=True):
    prepared=[]
    try:
        for path,label in [(front,'Front'),(back,'Back')]:prepared.append(prepare_photo(path,obj.name+' '+label,auto_crop,threshold))
    except Exception:
        for image,_,_ in prepared:
            if image.users==0:bpy.data.images.remove(image)
        raise
    coords=np.array([v.co[:] for v in obj.data.vertices]);lo=coords.min(0);hi=coords.max(0)
    # Template scan-line extents make a flat-lay's sleeves/body share the same
    # normalized width on each row, instead of projecting background as cloth.
    steps=512;ys=np.linspace(lo[1],hi[1],steps);mesh_bounds=[]
    edges=np.array([e.vertices[:] for e in obj.data.edges]);ea=coords[edges[:,0]];eb=coords[edges[:,1]]
    for y in ys:
        crossing=(np.minimum(ea[:,1],eb[:,1])<=y)&(np.maximum(ea[:,1],eb[:,1])>=y)&(abs(eb[:,1]-ea[:,1])>1e-6)
        a=ea[crossing];b=eb[crossing]
        if len(a):xs=a[:,0]+(y-a[:,1])/(b[:,1]-a[:,1])*(b[:,0]-a[:,0])
        else:xs=coords[np.argsort(abs(coords[:,1]-y))[:4],0]
        mesh_bounds.append((xs.min(),xs.max()))
    mesh_bounds=np.array(mesh_bounds)
    uv=obj.data.uv_layers.get('GS Photo UV') or obj.data.uv_layers.new(name='GS Photo UV')
    obj.data.uv_layers.active=uv
    obj.data.materials.clear()
    for image,_,_ in prepared:
        mat=bpy.data.materials.new(image.name+' fabric');mat.use_nodes=True
        bsdf=mat.node_tree.nodes.get('Principled BSDF');bsdf.inputs['Roughness'].default_value=.82
        tex=mat.node_tree.nodes.new('ShaderNodeTexImage');tex.image=image;tex.extension='EXTEND'
        uvnode=mat.node_tree.nodes.new('ShaderNodeUVMap');uvnode.uv_map='GS Photo UV'
        mat.node_tree.links.new(uvnode.outputs['UV'],tex.inputs['Vector'])
        mat.node_tree.links.new(tex.outputs['Color'],bsdf.inputs['Base Color'])
        obj.data.materials.append(mat)
    for poly in obj.data.polygons:
        # Split by local facing direction, not a global depth plane (which
        # mislabels curved shoulders). Back UVs read correctly from behind.
        side=0 if poly.normal.z>=0 else 1
        poly.material_index=side
        bounds=prepared[side][1]
        for loop_index in poly.loop_indices:
            p=coords[obj.data.loops[loop_index].vertex_index]
            v=float(np.clip((p[1]-lo[1])/max(hi[1]-lo[1],1e-6),0,1))
            a,b=mesh_bounds[min(round(v*(steps-1)),steps-1)]
            u=float(np.clip((p[0]-a)/max(b-a,1e-6),0,1))
            if side and flip_back:u=1-u
            left,right=bounds[min(round(v*(len(bounds)-1)),len(bounds)-1)]
            uv.data[loop_index].uv=(left+u*(right-left),v)
    obj.data.update()
    return [note for _,_,notes in prepared for note in notes]
