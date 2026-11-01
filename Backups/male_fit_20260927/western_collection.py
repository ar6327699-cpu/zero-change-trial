"""Three reference-photo Western sets on the existing Western rig/templates.

Original montage is packed unchanged. Fixed rest-space front/back UV panels
preserve lettering. Unseen seams are cloth samples, not invented graphics.
"""
import json,uuid,math
from pathlib import Path
import bpy
import numpy as np
import garment_templates as templates
import product_photo

ROOT=Path(__file__).resolve().parent.parent
REFERENCE=ROOT/'References/Western_20260927/three_sets_reference.png'
SETS=[
    dict(key='western_black_explore',name='Black Explore T-shirt + black cargo',
         shirt=[(49,78,245,400),(302,78,497,400)],sample=(124,333),
         pants=[(18,521,193,930),(336,521,518,930),(212,521,326,930)],
         waist=[(43,166),(371,489)],crotch=[(105,650),(424,647)],cargo=True),
    dict(key='western_white_brooklyn',name='White Brooklyn T-shirt + grey jeans',
         shirt=[(568,78,754,400),(797,78,981,400)],sample=(620,333),
         pants=[(546,521,714,930),(827,521,1006,930),(730,521,820,930)],
         waist=[(566,689),(859,984)],crotch=[(631,647),(918,649)],cargo=False),
    dict(key='western_beige_good_life',name='Beige Good Life T-shirt + black cargo',
         shirt=[(1061,78,1250,400),(1306,78,1490,400)],sample=(1142,333),
         pants=[(1033,521,1211,930),(1361,521,1528,930),(1227,521,1351,930)],
         waist=[(1060,1184),(1387,1500)],crotch=[(1129,649),(1444,646)],cargo=True,reuse_bottom='western_black_explore'),
]

def identifier(key,slot):return uuid.uuid5(uuid.NAMESPACE_URL,'fahad-western-20260927/'+key+'/'+slot).hex

def material(image,name,sample):
    m=bpy.data.materials.new(name);m.use_nodes=True
    n=m.node_tree.nodes;l=m.node_tree.links;b=n.get('Principled BSDF')
    b.inputs['Roughness'].default_value=.86;b.inputs['Specular IOR Level'].default_value=.16
    tex=n.new('ShaderNodeTexImage');tex.image=image;tex.extension='EXTEND'
    uv=n.new('ShaderNodeUVMap');uv.uv_map='GS Photo UV';l.new(uv.outputs[0],tex.inputs['Vector'])
    plain=n.new('ShaderNodeTexImage');plain.image=image
    point=n.new('ShaderNodeCombineXYZ');point.inputs['X'].default_value=sample[0]/image.size[0];point.inputs['Y'].default_value=1-sample[1]/image.size[1]
    l.new(point.outputs[0],plain.inputs['Vector'])
    attr=n.new('ShaderNodeVertexColor');attr.layer_name='GS_western_seam'
    mix=n.new('ShaderNodeMixRGB');l.new(attr.outputs['Color'],mix.inputs[0]);l.new(tex.outputs['Color'],mix.inputs[1]);l.new(plain.outputs['Color'],mix.inputs[2]);l.new(mix.outputs[0],b.inputs['Base Color'])
    return m

def finish_shirt(obj,design,image):
    product_photo.tailor(obj,'CREW',0.)
    co=np.array([v.co[:] for v in obj.data.vertices]);x,y,z=co.T
    body=np.clip((285-y)/35,0,1)*np.clip((45-abs(x))/12,0,1)
    # Slightly relaxed body and hip-length hem on NEW copies only.
    co[:,0]*=1+.08*body;co[:,1]+=9*np.clip((235-y)/49,0,1)
    obj.data.vertices.foreach_set('co',co.ravel());obj.data.update()
    band=co[(co[:,1]>235)&(co[:,1]<270)];half=np.percentile(abs(band[:,0]),98)
    low=co[:,1].min();high=co[:,1].max()
    uv=obj.data.uv_layers.get('GS Photo UV') or obj.data.uv_layers.new(name='GS Photo UV');obj.data.uv_layers.active=uv
    edge=obj.data.color_attributes.new(name='GS_western_seam',type='FLOAT_COLOR',domain='CORNER')
    obj.data.materials.clear()
    for side in ('Front','Back'):obj.data.materials.append(material(image,design['name']+' shirt '+side,design['sample']))
    for p in obj.data.polygons:
        back=p.normal.z<0;p.material_index=int(back);left,top,right,bottom=design['shirt'][int(back)]
        for li in p.loop_indices:
            v=obj.data.vertices[obj.data.loops[li].vertex_index];x,y,z=v.co
            rx=(-x if back else x)/half
            # Outer sleeves and unseen side seams use neutral cloth only.
            fade=max(np.clip((abs(rx)-.79)/.26,0,1),np.clip((.42-abs(v.normal.z))/.30,0,1))
            fade=fade*fade*(3-2*fade);edge.data[li].color=(fade,fade,fade,1)
            u=(left+right)/2+np.clip(rx,-.97,.97)*(right-left)/2
            sy=top+(high-y)/max(high-low,1)*(bottom-top)
            outline=np.interp(np.clip(rx,-1,1),[-1,-.7,-.36,0,.36,.7,1],[top+44,top+14,top+5,top+(24 if not back else 8),top+5,top+14,top+44])
            sy=float(np.clip(max(sy,outline),top+3,bottom-4))
            uv.data[li].uv=(u/image.size[0],1-sy/image.size[1])

def cross_sections(obj,coords,heights):
    edges=np.array([e.vertices[:] for e in obj.data.edges]);a=coords[edges[:,0]];b=coords[edges[:,1]];result=[]
    for y in heights:
        hit=(np.minimum(a[:,1],b[:,1])<=y)&(np.maximum(a[:,1],b[:,1])>=y)&(abs(a[:,1]-b[:,1])>1e-6)
        aa=a[hit];bb=b[hit];points=aa+(bb-aa)*((y-aa[:,1])/(bb[:,1]-aa[:,1]))[:,None]
        if len(points)<4:points=coords[np.argsort(abs(coords[:,1]-y))[:12]]
        sides=[]
        for sign in (-1,1):
            part=points[points[:,0]*sign>0]
            if not len(part):part=points
            sides.append((part[:,0].min(),part[:,0].max(),part[:,2].min(),part[:,2].max()))
        result.append(sides)
    return np.array(result)

def finish_pants(obj,design,image):
    if not design['cargo']:
        # Original Western shorts include separate cargo flaps and straps.
        # Jeans must not inherit them. Keep the continuous body component on
        # this NEW copy; protected Western/Eastern templates are untouched.
        import bmesh
        bm=bmesh.new();bm.from_mesh(obj.data);seen=set();components=[]
        for v in bm.verts:
            if v in seen:continue
            todo=[v];seen.add(v);component=[]
            while todo:
                a=todo.pop();component.append(a)
                for e in a.link_edges:
                    b=e.other_vert(a)
                    if b not in seen:seen.add(b);todo.append(b)
            components.append(component)
        main=max(components,key=len)
        bmesh.ops.delete(bm,geom=[v for component in components if component is not main for v in component],context='VERTS')
        bm.to_mesh(obj.data);bm.free();obj.data.update()
    co=np.array([v.co[:] for v in obj.data.vertices]);y=co[:,1];centre=np.where(co[:,0]>0,18.3,-18.3)
    # Existing relaxed trouser rig; cargo cuffs get a little extra ease.
    ease=1+(.12 if design['cargo'] else .025)*np.clip((135-y)/106,0,1)
    co[:,0]=centre+(co[:,0]-centre)*ease;co[:,2]=-2+(co[:,2]+2)*ease
    obj.data.vertices.foreach_set('co',co.ravel());obj.data.update()
    low,high=co[:,1].min(),co[:,1].max();heights=np.linspace(low+.001,high-.001,400);bounds=cross_sections(obj,co,heights)
    uv=obj.data.uv_layers.get('GS Photo UV') or obj.data.uv_layers.new(name='GS Photo UV');obj.data.uv_layers.active=uv
    edge=obj.data.color_attributes.new(name='GS_western_seam',type='FLOAT_COLOR',domain='CORNER')
    obj.data.materials.clear()
    left,top,right,bottom=design['pants'][0]
    sample=(left+(right-left)*.28,top+100)
    for label in ('Front','Back','Side'):obj.data.materials.append(material(image,design['name']+' pants '+label,sample))
    pixels=np.asarray(image.pixels[:],np.float32).reshape(image.size[1],image.size[0],4)[::-1]
    dark=pixels[:,:,:3].mean(2)<(.42 if design['cargo'] else .59)
    def cloth_bounds(panel,sy,half=None):
        left,top,right,bottom=design['pants'][panel]
        iy=int(np.clip(round(sy),top+4,bottom-8))
        lo,hi=left+3,right-3
        if half is not None:
            middle=design['crotch'][panel][0]
            if half<0:hi=middle-3
            else:lo=middle+3
        for offset in [0,-1,1,-3,3,-6,6,-12,12]:
            yy=int(np.clip(iy+offset,top+4,bottom-8));ids=np.flatnonzero(dark[yy,lo:hi])+lo
            if len(ids)>12:
                inset=min(6.,(ids[-1]-ids[0])*.12)
                return float(ids[0]+inset),float(ids[-1]-inset)
        return float(lo+5),float(hi-5)
    for p in obj.data.polygons:
        center=sum((obj.data.vertices[i].co for i in p.vertices),__import__('mathutils').Vector())/len(p.vertices)
        # Fixed outer-side panel for the provided pocket view; no per-frame
        # material switching. Inner leg surfaces remain front/back cloth.
        is_side=abs(p.normal.x)>.73 and center.x*p.normal.x>0
        index=2 if is_side else (0 if p.normal.z>=0 else 1);p.material_index=index
        left,top,right,bottom=design['pants'][index]
        for li in p.loop_indices:
            vi=obj.data.loops[li].vertex_index;x,y,z=co[vi];sign=1 if x>=0 else -1
            hidx=int(np.clip(round((y-low)/(high-low)*399),0,399));xmin,xmax,zmin,zmax=bounds[hidx,1 if sign>0 else 0]
            t=np.clip((high-y)/(high-low),0,1);sy=top+t*(bottom-top-5)
            if is_side:
                sy=min(sy,bottom-28)  # Do not project the model's shoes onto cuffs.
                lo,hi=cloth_bounds(index,sy)
                u=lo+(1-np.clip((z-zmin)/max(zmax-zmin,1),0,1))*(hi-lo)
            else:
                back=index==1;ix=-x if back else x
                full=max(abs(bounds[hidx,:,0].min()),abs(bounds[hidx,:,1].max()),1)
                outer_left,outer_right=left+5,right-5;wa,wb=design['waist'][index];cx,cy=design['crotch'][index]
                # Waist/seat is one panel, then each leg maps independently.
                blend=np.clip((sy-(cy-25))/45,0,1);blend=blend*blend*(3-2*blend)
                wl,wr=cloth_bounds(index,sy)
                upper=(wl+wr)/2+ix/full*(wr-wl)/2
                f=np.clip((x-xmin)/max(xmax-xmin,1),0,1)
                if back:f=1-f
                lo,hi=cloth_bounds(index,sy,-1 if ix<0 else 1)
                lower=lo+f*(hi-lo);u=upper*(1-blend)+lower*blend
            uv.data[li].uv=(float(u/image.size[0]),float(1-sy/image.size[1]))
            # Soften only extreme grazing inner seams, preserving side pockets.
            fade=0. if is_side else np.clip((abs(obj.data.vertices[vi].normal.x)-.85)/.15,0,1)*.6
            edge.data[li].color=(fade,fade,fade,1)

def build(directory):
    directory=Path(directory);directory.mkdir(exist_ok=True,parents=True)
    image=bpy.data.images.load(str(REFERENCE),check_existing=True);image.name='Western three sets original reference';image.pack()
    rows=[];outfits=[]
    for design in SETS:
        ids={}
        for slot,category in [('top','HALF_SHIRT'),('bottom','TROUSERS')]:
            if slot=='bottom' and design.get('reuse_bottom'):
                ids[slot]=identifier(design['reuse_bottom'],'bottom')
                continue
            identifier_=identifier(design['key'],slot);ids[slot]=identifier_
            source=bpy.data.objects[templates.PREFIX+category]
            obj=templates.clone(source,'GS Item '+identifier_,templates.storage())
            if obj.data.shape_keys:obj.shape_key_clear()
            for mod in list(obj.modifiers):
                if mod.type!='ARMATURE':obj.modifiers.remove(mod)
            obj.data.materials.clear()
            for prop in list(obj.keys()):
                if prop.startswith(('gs_','AR_')):del obj[prop]
            if slot=='top':finish_shirt(obj,design,image)
            else:finish_pants(obj,design,image)
            obj['gs_id']=identifier_;obj['gs_category']=category;obj['gs_profile']='MALE';obj['gs_collection']='WESTERN'
            obj['gs_label']=design['name'].split(' + ')[0 if slot=='top' else 1]
            obj['gs_photo_method']='Western reference atlas v1: fixed front/back and trouser side panels'
            obj['gs_reference']=str(REFERENCE);obj['gs_western_version']=1
            bpy.data.libraries.write(str(directory/(identifier_+'.blend')),{obj},fake_user=True)
            rows.append({'id':identifier_,'name':obj['gs_label'],'category':category,'profile':'MALE','collection':'WESTERN','photo_layout':'PRODUCT','method':obj['gs_photo_method'],
                'notes':['Original reference artwork preserved. Existing Western rig and calibration. Photo-based appearance; not a sewn-pattern reconstruction.'],
                'source_regions':design['shirt'] if slot=='top' else design['pants']})
        outfits.append({'key':design['key'],'name':design['name'],'profile':'MALE','collection':'WESTERN',**ids})
    (directory/'library_additions.json').write_text(json.dumps(rows,indent=2))
    (directory/'outfit_additions.json').write_text(json.dumps(outfits,indent=2))
    return rows,outfits
