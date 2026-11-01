"""Fixed rest-space sleeve charts from the existing product photographs.

Only Female photo-pattern assets participate. Central artwork is untouched;
unknown side seams blend into the material's existing sampled cloth colour.
No generated textures and no view-dependent UVs.
"""
import json
import bpy
import numpy as np
from collections import deque
from pathlib import Path
import eastern_photos as ep

VERSION=1

def sleeve_chart(image,profile):
    path=Path(bpy.path.abspath(image.filepath))
    if not path.is_file():return None
    original=bpy.data.images.load(str(path),check_existing=False)
    try:
        w,h=original.size
        a=np.asarray(original.pixels[:],np.float32).reshape(h,w,4)[::-1,:,:3]
        left,up,right,bottom=profile.get('source_region',(0,0,1,1))
        step=2 if (right-left)*w<600 else 4
        a=a[round(up*h):round(bottom*h):step,round(left*w):round(right*w):step]
    finally:bpy.data.images.remove(original)
    # Do not close the narrow slit between the sleeve and torso: morphology
    # joins the strips and sends the chart back into the torso artwork.
    mh,mw=a.shape[:2];chroma=a.max(2)-a.min(2)
    top,bottom=profile['top'],profile['bottom'];rows=[]
    # Dark green cloth has lower linear-RGB chroma. Try conservative lower
    # thresholds only if the original threshold cannot identify both sleeves.
    for fill,threshold in ((g,t) for g in (False,True) for t in (.09,.055,.03)):
        mask=chroma>threshold;rows=[]
        if fill:
            # Fill enclosed neutral embroidery, but leave background channels
            # between sleeves and torso connected to the image exterior.
            outside=np.zeros_like(mask);pending=deque()
            for yy,xx in ([(0,x) for x in range(mw)]+[(mh-1,x) for x in range(mw)]+
                          [(y,0) for y in range(mh)]+[(y,mw-1) for y in range(mh)]):
                if not mask[yy,xx] and not outside[yy,xx]:
                    outside[yy,xx]=True;pending.append((yy,xx))
            while pending:
                yy,xx=pending.popleft()
                for ny,nx in ((yy-1,xx),(yy+1,xx),(yy,xx-1),(yy,xx+1)):
                    if 0<=ny<mh and 0<=nx<mw and not mask[ny,nx] and not outside[ny,nx]:
                        outside[ny,nx]=True;pending.append((ny,nx))
            mask=~outside
        for y in range(int((top+.16*(bottom-top))*mh),int((top+.68*(bottom-top))*mh)):
            runs=[r for r in ep.runs(mask[y]) if len(r)>=4]
            if len(runs)!=3:continue
            if not runs[1][0]<profile['centre']*mw<runs[1][-1]:continue
            rows.append((y/mh,(runs[0][0]+runs[0][-1])/(2*mw),len(runs[0])/mw,
                         (runs[2][0]+runs[2][-1])/(2*mw),len(runs[2])/mw))
        if len(rows)<8:continue
        candidate=np.array(rows);width=np.median(candidate[:,[2,4]],axis=0)
        rows=candidate[np.all(candidate[:,[2,4]]>width*.65,axis=1)]
        if len(rows)>=8:break
    if len(rows)<8:return None
    rows=np.array(rows)
    end=float(rows[-1,0]);start=top+.075*(bottom-top)
    # Smooth trajectories suppress row-to-row mask/embroidery gaps. Cuffs are
    # located from separated sleeve strips, not the torso's lower silhouette.
    centres=[np.polyfit(rows[:,0],rows[:,j],1) for j in (1,3)]
    return start,end,centres,width

def revise(obj):
    if obj.get('gs_category')!='FEMALE_KAMEEZ' or not obj.get('gs_photo_profiles'):return False
    if obj.data.get('gs_female_finish_revision')==VERSION:return False
    profiles=json.loads(obj['gs_photo_profiles']);uv=obj.data.uv_layers.get('GS Photo UV')
    regions=obj.data.attributes.get('GS_pattern_region')
    if uv is None or regions is None:return False
    c=np.array([v.co[:] for v in (obj.data.shape_keys.key_blocks[0].data if obj.data.shape_keys else obj.data.vertices)])
    edge=obj.data.color_attributes.get('GS_side_blend')
    charts=[]
    for side,m in enumerate(obj.data.materials[:2]):
        tex=m.node_tree.nodes.get('Image Texture') if m and m.use_nodes else None
        charts.append(sleeve_chart(tex.image,profiles[side]) if tex and tex.image else None)
    for poly in obj.data.polygons:
        side=min(poly.material_index,1);region=regions.data[poly.index].value
        for li in poly.loop_indices:
            i=obj.data.loops[li].vertex_index;x,y,z=c[i]
            if region and charts[side] is not None:
                start,end,centres,width=charts[side]
                sign=1 if x>0 else -1;photo_sign=-sign if side else sign;column=1 if photo_sign>0 else 0
                sleeve=c[(c[:,0]*sign>45)&(c[:,1]>270)]
                t=float(np.clip((abs(x)-40)/max(np.max(abs(sleeve[:,0]))-40,1),0,1))
                ring=c[(abs(c[:,0]-x)<.02)&(c[:,1]>270)]
                cy=(ring[:,1].max()+ring[:,1].min())*.5
                radius=max(np.ptp(ring[:,1])*.5,1.)
                # Cylindrical arc length, not a cosine projection whose UV
                # derivative collapses at the visible outer sleeve seam.
                around=float(np.arcsin(np.clip((y-cy)/radius,-1,1))*2/np.pi)
                v=start+(end-start)*t
                u=float(np.polyval(centres[column],v)+photo_sign*around*width[column]*.43)
                uv.data[li].uv=ep.source_uv(u,1-v,profiles[side])
            elif not region and edge is not None:
                # Fade clamped image edges before their tiny UV area gets
                # stretched across a broad side panel. Neck/chest stay intact.
                ring=c[(abs(c[:,1]-y)<2)&(abs(c[:,0])<65)]
                if len(ring)<8 or y>280:continue
                radius=max(np.max(abs(ring[:,0])),1.)
                t=float(np.clip((abs(x)/radius-.70)/.22,0,1));t=t*t*(3-2*t)
                old=edge.data[li].color[0];blend=max(old,t)
                edge.data[li].color=(blend,blend,blend,1.)
    obj.data['gs_female_finish_revision']=VERSION;obj.data.update()
    return True
