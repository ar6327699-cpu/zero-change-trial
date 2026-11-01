"""Photo-to-template garment library, isolated from the known-good AR folder."""
import bpy,json,uuid,re,os
from pathlib import Path
from mathutils import Matrix,Vector,Quaternion
import math
import garment_templates as templates
import garment_photos as photos
import product_photo

ROOT=Path(__file__).resolve().parent.parent
LIBRARY=ROOT/'Garments'
API=None
ENUM_CACHE=[]
PREVIEW='GS Garment Preview'

def scene():
    found=[s for s in bpy.data.scenes if s.get('GS_studio_version')]
    if len(found)!=1:raise RuntimeError('Open the separate AR Garment Studio project using its launcher')
    return found[0]

def guard(stopped=True):
    if Path(bpy.data.filepath).resolve()!=ROOT/'fahadGenius_AR_experiment.blend':
        raise RuntimeError('Garment tools are restricted to the separate Studio copy')
    s=scene();session=bpy.app.driver_namespace.get('fahad_ar_runtime')
    if stopped and session and session.running:
        raise RuntimeError('Stop AR Camera before building, previewing, or changing garments')
    return s

def manifest():
    file=LIBRARY/'library.json'
    if not file.exists():return []
    rows=json.loads(file.read_text(encoding='utf-8'))
    if not isinstance(rows,list):raise ValueError('Garment library index is invalid')
    for r in rows:
        if r['category'] not in templates.CATEGORIES or not re.fullmatch('[0-9a-f]{32}',r['id']):raise ValueError('Invalid garment library entry')
    return rows

def choices(self,context):
    global ENUM_CACHE
    try:ENUM_CACHE=[(r['id'],r['name']+' — '+templates.CATEGORIES[r['category']][0],'Reviewed photo template') for r in manifest()]
    except Exception:ENUM_CACHE=[]
    if not ENUM_CACHE:ENUM_CACHE=[('NONE','No garments yet','Build a garment from front/back photos')]
    return ENUM_CACHE

def item(identifier):
    row=next((r for r in manifest() if r['id']==identifier),None)
    if not row:raise ValueError('Select a garment from the library first')
    name='GS Item '+identifier;obj=bpy.data.objects.get(name)
    if obj is None:
        file=LIBRARY/(identifier+'.blend')
        if not file.is_file():raise ValueError('Saved garment asset is missing')
        with bpy.data.libraries.load(str(file),link=False) as (src,dst):
            if name not in src.objects:raise ValueError('Saved garment asset does not contain the expected object')
            dst.objects=[name]
        obj=dst.objects[0];templates.storage().objects.link(obj)
        # Library loading must not switch the live armature to an appended rig.
        obj.parent=bpy.data.objects['Armature']
        for m in obj.modifiers:
            if m.type=='ARMATURE':m.object=obj.parent
        obj.hide_render=True;obj.hide_set(True)
    return obj,row

def build(name,category,front,back,auto=True,threshold=.16,flip_back=True,photo_layout='FLAT',neckline='AUTO',slim=.08):
    guard();templates.ensure_templates()
    if category not in templates.CATEGORIES:raise ValueError('Unknown garment category')
    identifier=uuid.uuid4().hex;obj=None
    try:
        source=bpy.data.objects[templates.PREFIX+category]
        obj=templates.clone(source,'GS Item '+identifier,templates.storage())
        if obj.data.shape_keys:obj.shape_key_clear()
        # Landmark attributes/modifiers are regenerated only on activation.
        for mod in list(obj.modifiers):
            if mod.type!='ARMATURE':obj.modifiers.remove(mod)
        if photo_layout=='PRODUCT' and category in ('HALF_SHIRT','FULL_SHIRT'):
            notes=product_photo.apply(obj,front,back,neckline,slim,flip_back)
        else:notes=photos.apply_photos(obj,front,back,auto,threshold,flip_back)
        obj['gs_category']=category;obj['gs_label']=name.strip()[:80] or 'New garment';obj['gs_id']=identifier
        row={'id':identifier,'name':obj['gs_label'],'category':category,'notes':notes,
             'method':obj.get('gs_photo_method','rigged template + local front/back photo projection; not exact 3D reconstruction'),
             'photo_layout':photo_layout,'neckline':obj.get('gs_neckline','template'),'slim':obj.get('gs_slim',0)}
        LIBRARY.mkdir(exist_ok=True)
        bpy.data.libraries.write(str(LIBRARY/(identifier+'.blend')),{obj},fake_user=True)
        rows=manifest();rows.append(row)
        temp=LIBRARY/'library.json.tmp';temp.write_text(json.dumps(rows,indent=2),encoding='utf-8');os.replace(temp,LIBRARY/'library.json')
        return obj,row
    except Exception:
        if obj:bpy.data.objects.remove(obj,do_unlink=True)
        raise

def return_to_ar():
    s=scene()
    if bpy.context.window:bpy.context.window.scene=s
    return s

def prepare_active(slot,source,category):
    active=bpy.data.objects[slot]
    # Runtime-only waistband weights are regenerated for each replacement;
    # they are not part of the template's rig-group compatibility contract.
    import waist_layer
    runtime_group=active.vertex_groups.get(waist_layer.GROUP)
    active_groups=[g.name for g in active.vertex_groups if g!=runtime_group]
    if active_groups!=[g.name for g in source.vertex_groups]:
        raise ValueError('Bone-group order differs; refusing unsafe garment replacement')
    if runtime_group:active.vertex_groups.remove(runtime_group)
    active.data=source.data.copy()
    if category:active['gs_category']=category
    elif 'gs_category' in active:del active['gs_category']
    active['gs_label']=source.get('gs_label','Original '+slot)
    for mod in list(active.modifiers):
        if mod.type=='NODES' and mod.name=='AR Landmark Fit':active.modifiers.remove(mod)
    active.hide_render=True;active.hide_set(True)

def activate(identifier):
    s=guard();obj,row=item(identifier);return_to_ar()
    category=row['category'];slot=templates.CATEGORIES[category][1]
    active=bpy.data.objects[slot]
    previous=(active.data,active.get('gs_category'),active.get('gs_label'))
    try:
        prepare_active(slot,obj,category)
        if slot=='Tops':API.prepare_shirt()
        API.garment_fit.GarmentFitter(bpy.data.objects['Armature'])
    except Exception:
        active.data=previous[0]
        for key,value in zip(['gs_category','gs_label'],previous[1:]):
            if value is None:
                if key in active:del active[key]
            else:active[key]=value
        API.garment_fit.GarmentFitter(bpy.data.objects['Armature'])
        raise
    if slot=='Bottoms':s.far_outfit='OUTFIT'
    s.gs_status='Active: '+row['name']+'. Start AR Camera and calibrate for the wearer.'

def restore():
    s=guard();return_to_ar()
    for slot in ['Tops','Bottoms']:prepare_active(slot,bpy.data.objects['GS Original '+slot],None)
    API.prepare_shirt();API.garment_fit.GarmentFitter(bpy.data.objects['Armature'])
    s.gs_status='Original outfit restored in Studio. Known-good AR folder never changed.'

def preview(identifier):
    guard();obj,row=item(identifier)
    preview_scene=bpy.data.scenes.get(PREVIEW)
    if preview_scene is None:
        preview_scene=bpy.data.scenes.new(PREVIEW)
        preview_scene.world=bpy.data.worlds.new('GS Preview World');preview_scene.world.color=(.12,.12,.12)
        camera=bpy.data.objects.new('GS Preview Camera',bpy.data.cameras.new('GS Preview Camera'))
        preview_scene.collection.objects.link(camera);preview_scene.camera=camera
        camera.data.type='ORTHO';camera.data.ortho_scale=4.6
        preview_scene.render.engine='CYCLES';preview_scene.cycles.samples=16
        preview_scene.render.resolution_x=960;preview_scene.render.resolution_y=720
        for name,location,power in [('GS Preview Key',(-3,-4,6),1000),('GS Preview Fill',(3,2,4),700)]:
            lamp=bpy.data.lights.new(name,'AREA');lamp.energy=power;lamp.shape='DISK';lamp.size=5
            light=bpy.data.objects.new(name,lamp);preview_scene.collection.objects.link(light);light.location=location
            light.rotation_euler=(Vector((0,0,1.6))-light.location).to_track_quat('-Z','Y').to_euler()
    old=bpy.data.objects.get('GS Preview Garment')
    if old:bpy.data.objects.remove(old,do_unlink=True)
    view=obj.copy();view.name='GS Preview Garment';view.data=obj.data
    preview_scene.collection.objects.link(view);view.parent=None;view.modifiers.clear();view.matrix_world=API.BASE.copy()
    view.hide_viewport=False;view.hide_render=False;view.hide_set(False)
    previous_rig=bpy.data.objects.get('GS Preview Rig')
    if previous_rig:bpy.data.objects.remove(previous_rig,do_unlink=True)
    if obj.get('gs_photo_method') and scene().gs_relaxed_preview:
        original=bpy.data.objects['Armature'];rig=original.copy();rig.data=original.data.copy();rig.name='GS Preview Rig'
        rig.animation_data_clear();preview_scene.collection.objects.link(rig);rig.parent=None;rig.matrix_world=API.BASE.copy();rig.data.pose_position='POSE'
        rig.hide_viewport=False;rig.hide_render=False;rig.hide_set(False)
        for b in rig.pose.bones:b.matrix_basis=Matrix.Identity(4)
        for side,sign in [('Left',1),('Right',-1)]:
            b=rig.pose.bones['mixamorig:'+side+'Arm'];rest=b.bone.matrix_local.to_quaternion()
            b.rotation_mode='QUATERNION';b.rotation_quaternion=rest.inverted()@Quaternion((0,0,1),-sign*math.radians(62))@rest
        view.parent=rig;view.matrix_parent_inverse=Matrix.Identity(4);view.matrix_basis=Matrix.Identity(4)
        view.modifiers.new('Preview skinning','ARMATURE').object=rig
    if bpy.context.window:bpy.context.window.scene=preview_scene
    bpy.context.view_layer.update()
    evaluated=view.evaluated_get(bpy.context.evaluated_depsgraph_get())
    points=[view.matrix_world@v.co for v in evaluated.data.vertices]
    lo=Vector(tuple(min(p[k] for p in points) for k in range(3)));hi=Vector(tuple(max(p[k] for p in points) for k in range(3)))
    preview_scene['gs_focus']=tuple((lo+hi)*.5)
    aspect=preview_scene.render.resolution_x/preview_scene.render.resolution_y
    preview_scene.camera.data.ortho_scale=max(hi.x-lo.x,hi.y-lo.y,(hi.z-lo.z)*aspect)*1.25
    if bpy.context.window:bpy.context.window.scene=preview_scene
    preview_angle('FRONT')
    scene().gs_status='Preview only: '+row['name']+'. Check front, back, sides; then Activate selected.'

def preview_angle(angle):
    sc=bpy.data.scenes.get(PREVIEW)
    if not sc:raise ValueError('Preview a garment first')
    camera=sc.camera
    target=Vector(sc.get('gs_focus',(0,0,1.7)))
    camera.location=target+Vector({'FRONT':(0,-8,0),'BACK':(0,8,0),'LEFT':(8,0,0),'RIGHT':(-8,0,0)}[angle])
    camera.rotation_euler=(target-camera.location).to_track_quat('-Z','Y').to_euler()
    if bpy.context.screen:
        for area in bpy.context.screen.areas:
            if area.type=='VIEW_3D':
                area.spaces.active.region_3d.view_perspective='CAMERA'
                area.spaces.active.shading.type='MATERIAL'

class GS_OT_build(bpy.types.Operator):
    bl_idname='garment_studio.build';bl_label='Build photo garment';bl_description='Save a new library garment; existing outfit is not replaced'
    def execute(self,context):
        try:
            s=guard();obj,row=build(s.gs_name,s.gs_category,s.gs_front,s.gs_back,s.gs_auto_crop,s.gs_threshold,s.gs_flip_back,s.gs_layout,s.gs_neckline,s.gs_slim)
            s.gs_selected=row['id'];s.gs_status='Built '+row['name']+'. Preview before activating.'
            if row['notes']:s.gs_status+=' '+row['notes'][0]
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        return {'FINISHED'}

class GS_OT_preview(bpy.types.Operator):
    bl_idname='garment_studio.preview';bl_label='Preview selected (no replacement)'
    def execute(self,context):
        try:preview(scene().gs_selected)
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        return {'FINISHED'}

class GS_OT_angle(bpy.types.Operator):
    bl_idname='garment_studio.angle';bl_label='View garment'
    angle:bpy.props.EnumProperty(items=[(x,x.title(),'') for x in ['FRONT','BACK','LEFT','RIGHT']])
    def execute(self,context):
        preview_angle(self.angle);return {'FINISHED'}

class GS_OT_activate(bpy.types.Operator):
    bl_idname='garment_studio.activate';bl_label='Activate selected for AR'
    def execute(self,context):
        try:activate(scene().gs_selected)
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        return {'FINISHED'}

class GS_OT_restore(bpy.types.Operator):
    bl_idname='garment_studio.restore';bl_label='Restore original shirt + shorts'
    def execute(self,context):
        try:restore()
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}
        return {'FINISHED'}

class GS_OT_return(bpy.types.Operator):
    bl_idname='garment_studio.return_ar';bl_label='Back to AR (keep outfit)'
    def execute(self,context):
        return_to_ar();return {'FINISHED'}

class GS_PT_panel(bpy.types.Panel):
    bl_label='Garment Studio - SAFE COPY';bl_idname='GS_PT_panel'
    bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='Garments'
    def draw(self,context):
        l=self.layout
        try:s=scene()
        except Exception:l.label(text='Launch the separate Garment Studio copy');return
        l.label(text='Offline photo-to-template importer')
        l.label(text='Original AR Baised folder is untouched')
        if context.scene.name==PREVIEW:
            row=l.row()
            for x in ['FRONT','BACK','LEFT','RIGHT']:row.operator('garment_studio.angle',text=x.title()).angle=x
            l.operator('garment_studio.return_ar')
            l.operator('garment_studio.activate')
            l.prop(s,'gs_relaxed_preview');l.operator('garment_studio.preview',text='Refresh preview pose')
        box=l.box();box.label(text='1. Front + back garment photos')
        box.prop(s,'gs_name');box.prop(s,'gs_category');box.prop(s,'gs_front');box.prop(s,'gs_back')
        if s.gs_category in ('HALF_SHIRT','FULL_SHIRT'):box.prop(s,'gs_layout')
        if s.gs_layout=='PRODUCT' and s.gs_category in ('HALF_SHIRT','FULL_SHIRT'):
            box.prop(s,'gs_neckline');box.prop(s,'gs_slim')
            box.label(text='Product photo: centred torso, no row stretching')
        else:
            box.prop(s,'gs_auto_crop')
            if s.gs_auto_crop:box.prop(s,'gs_threshold')
        box.prop(s,'gs_flip_back')
        if s.gs_layout=='PRODUCT' and s.gs_category in ('HALF_SHIRT','FULL_SHIRT'):
            box.label(text='Whole garment visible; plain background')
        else:box.label(text='Flat, straight, sleeves spread; plain background')
        box.operator('garment_studio.build')
        box=l.box();box.label(text='2. Library - inspect before replacing')
        box.prop(s,'gs_selected',text='Garment');box.operator('garment_studio.preview');box.operator('garment_studio.activate')
        box=l.box();box.label(text='3. Fahad AR tab > Start AR Camera')
        box.label(text='Calibrate for wearer; hips/knees/ankles visible')
        box.label(text='Stop camera before switching garments')
        box.operator('garment_studio.restore')
        for slot in ['Tops','Bottoms']:box.label(text=slot+': '+bpy.data.objects[slot].get('gs_label','Original'))
        l.label(text=s.gs_status)
        l.label(text='Template approximation, not exact tailoring')

CLASSES=[GS_OT_build,GS_OT_preview,GS_OT_angle,GS_OT_activate,GS_OT_restore,GS_OT_return,GS_PT_panel]
def register(ar):
    global API
    API=ar;guard(stopped=False);templates.ensure_templates()
    for cls in CLASSES:
        old=getattr(bpy.types,cls.__name__,None)
        if old:bpy.utils.unregister_class(old)
        bpy.utils.register_class(cls)
    properties={'gs_name':bpy.props.StringProperty(name='Garment name',default='My garment'),
        'gs_category':bpy.props.EnumProperty(name='Template',items=[(k,v[0],v[0]+' rigged template') for k,v in templates.CATEGORIES.items()]),
        'gs_front':bpy.props.StringProperty(name='Front photo',subtype='FILE_PATH'),
        'gs_back':bpy.props.StringProperty(name='Back photo',subtype='FILE_PATH'),
        'gs_layout':bpy.props.EnumProperty(name='Photo layout',items=[('PRODUCT','Product photo / arms down','Anchored chest/logo mapping and sleeve projection'),('FLAT','Flat-lay / T-pose','Original flat-lay scan-line mapping')],default='PRODUCT'),
        'gs_neckline':bpy.props.EnumProperty(name='Neckline',items=[('AUTO','Auto estimate','Estimate crew vs V; inspect preview'),('CREW','Round / crew neck','Rounded neck opening'),('V','Original V-neck','Keep original template neck')],default='AUTO'),
        'gs_slim':bpy.props.FloatProperty(name='Torso slimming',default=.08,min=0,max=.16,subtype='FACTOR'),
        'gs_relaxed_preview':bpy.props.BoolProperty(name='Relaxed arms (preview only)',default=True),
        'gs_auto_crop':bpy.props.BoolProperty(name='Auto crop plain background',default=True),
        'gs_threshold':bpy.props.FloatProperty(name='Background tolerance',default=.16,min=.04,max=.5),
        'gs_flip_back':bpy.props.BoolProperty(name='Back photo viewed from behind',default=True),
        'gs_selected':bpy.props.EnumProperty(name='Garment',items=choices),
        'gs_status':bpy.props.StringProperty(default='Ready. Add front/back garment photos.')}
    for name,prop in properties.items():setattr(bpy.types.Scene,name,prop)
