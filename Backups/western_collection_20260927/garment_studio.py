"""Photo-to-template garment library, isolated from the known-good AR folder."""
import bpy,json,uuid,re,os
from pathlib import Path
from mathutils import Matrix,Vector,Quaternion
import math
import garment_templates as templates
import garment_photos as photos
import product_photo
import garment_visibility as visibility
import eastern_master
import eastern_photos
import female_patterns
import body_profiles

ROOT=Path(__file__).resolve().parent.parent
LIBRARY=ROOT/'Garments'
API=None
ENUM_CACHE=[]
CATEGORY_CACHE=[]
OUTFIT_CACHE=[]
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
    try:ENUM_CACHE=[('NONE','Select garment','Choose an outfit for this profile')]+[(r['id'],r['name']+' — '+templates.CATEGORIES[r['category']][0],'Photo garment: inspect before try-on') for r in manifest() if r.get('profile','MALE')==body_profiles.current()]
    except Exception:ENUM_CACHE=[]
    if not ENUM_CACHE:ENUM_CACHE=[('NONE','No garments yet','Build a garment from front/back photos')]
    return ENUM_CACHE

def categories(self,context):
    global CATEGORY_CACHE
    profile=body_profiles.current()
    CATEGORY_CACHE=[(k,v[0],v[0],i) for i,(k,v) in enumerate(templates.CATEGORIES.items()) if (k in female_patterns.CATEGORIES)==(profile=='FEMALE')]
    return CATEGORY_CACHE

def outfits():
    path=LIBRARY/'outfits.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else []

def outfit_choices(self,context):
    global OUTFIT_CACHE
    OUTFIT_CACHE=[('NONE','Select complete suit','Kameez / Kurta with matching lower')]+[(r['key'],r['name'],'Stored front/back photo outfit') for r in outfits() if r['profile']==body_profiles.current()]
    return OUTFIT_CACHE

def activate_outfit(key):
    s=guard();row=next((r for r in outfits() if r['key']==key),None)
    if not row or row['profile']!=body_profiles.current():raise ValueError('Choose a suit in the active profile section')
    # Validate both assets before replacing either actor.
    item(row['top']);item(row['bottom'])
    previous={n:(bpy.data.objects[n].data,dict(bpy.data.objects[n].items())) for n in ('Tops','Bottoms')}
    try:
        activate(row['top']);activate(row['bottom'])
        import accessories
        accessories.activate(row.get('dupatta'))
    except Exception:
        for name,(mesh,properties) in previous.items():
            obj=bpy.data.objects[name];obj.data=mesh
            for prop in ('gs_category','gs_label','gs_profile','gs_id'):
                if prop in properties:obj[prop]=properties[prop]
                elif prop in obj:del obj[prop]
        API.prepare_shirt();API.garment_fit.GarmentFitter(bpy.data.objects['Armature']);body_profiles.capture()
        raise
    s.gs_status='Active suit: '+row['name']+'. Verify calibration before try-on.'

def item(identifier):
    row=next((r for r in manifest() if r['id']==identifier),None)
    if not row:raise ValueError('Select a garment from the library first')
    name='GS Item '+identifier;obj=bpy.data.objects.get(name)
    if obj is None:
        file=LIBRARY/(identifier+'.blend')
        if not file.is_file():raise ValueError('Saved garment asset is missing')
        objects_before=set(bpy.data.objects)
        with bpy.data.libraries.load(str(file),link=False) as (src,dst):
            if name not in src.objects:raise ValueError('Saved garment asset does not contain the expected object')
            dst.objects=[name]
        obj=dst.objects[0]
        # Library loading must not switch the live armature to an appended rig.
        obj.parent=bpy.data.objects['Armature']
        for m in obj.modifiers:
            if m.type=='ARMATURE':m.object=obj.parent
        # Appending a garment also appends its saved parent rig dependency.
        # Once rebound to the shared rig, retire ONLY new, unlinked orphans.
        for imported in set(bpy.data.objects)-objects_before:
            if imported!=obj and imported.type=='ARMATURE' and imported.users==0:
                data=imported.data;bpy.data.objects.remove(imported)
                if data.users==0:bpy.data.armatures.remove(data)
    import cosmetic_finish
    cosmetic_finish.finish_top(obj)
    visibility.store(obj,templates.storage())
    return obj,row

def build(name,category,front,back,auto=True,threshold=.16,flip_back=True,photo_layout='FLAT',neckline='AUTO',slim=.08,regions=None):
    guard();templates.ensure_templates()
    if category not in templates.CATEGORIES:raise ValueError('Unknown garment category')
    identifier=uuid.uuid4().hex;obj=None
    try:
        source=(female_patterns.ensure(category,templates.storage()) if category in female_patterns.CATEGORIES else eastern_master.ensure(category,templates.storage(),templates.clone)
                if category in templates.eastern.CATEGORIES else bpy.data.objects[templates.PREFIX+category])
        obj=templates.clone(source,'GS Item '+identifier,templates.storage())
        if obj.data.shape_keys:obj.shape_key_clear()
        # Landmark attributes/modifiers are regenerated only on activation.
        for mod in list(obj.modifiers):
            if mod.type!='ARMATURE':obj.modifiers.remove(mod)
        if category in female_patterns.CATEGORIES:
            import female_photos
            notes=female_photos.apply(obj,front,back,category,flip_back,regions)
            female_patterns.regional_keys(obj)
        elif photo_layout=='PRODUCT' and category in templates.eastern.CATEGORIES:
            notes=eastern_photos.apply(obj,front,back,category,flip_back,regions=regions)
        elif photo_layout=='PRODUCT' and category in ('HALF_SHIRT','FULL_SHIRT'):
            notes=product_photo.apply(obj,front,back,neckline,slim,flip_back)
        else:notes=photos.apply_photos(obj,front,back,auto,threshold,flip_back)
        obj['gs_category']=category;obj['gs_label']=name.strip()[:80] or 'New garment';obj['gs_id']=identifier
        obj['gs_profile']='FEMALE' if category in female_patterns.CATEGORIES else 'MALE'
        row={'id':identifier,'name':obj['gs_label'],'category':category,'notes':notes,
             'method':obj.get('gs_photo_method','rigged template + local front/back photo projection; not exact 3D reconstruction'),
             'photo_layout':photo_layout,'neckline':obj.get('gs_neckline','template'),'slim':obj.get('gs_slim',0),
             'profile':obj['gs_profile'],'source_regions':regions}
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
    if runtime_group:active.vertex_groups.remove(runtime_group)
    if active_groups!=[g.name for g in source.vertex_groups] and category not in female_patterns.CATEGORIES and active.get('gs_profile')!='FEMALE':
        templates.eastern.prepare_groups(active,source,category)
    active.data=source.data.copy()
    if category in female_patterns.CATEGORIES:
        # Female patterns are authored for linear skinning. Inheriting the
        # Male actor's volume-preserving mode balloons broad elbow blends.
        for modifier in active.modifiers:
            if modifier.type=='ARMATURE':modifier.use_deform_preserve_volume=False
    active.data['gs_profile_runtime_copy']=True
    active['gs_profile']=source.get('gs_profile','MALE')
    if 'gs_id' in source:active['gs_id']=source['gs_id']
    elif 'gs_id' in active:del active['gs_id']
    if 'gs_clean_reference_version' in source:active['gs_clean_reference_version']=source['gs_clean_reference_version']
    elif 'gs_clean_reference_version' in active:del active['gs_clean_reference_version']
    import accessories
    accessories.surface(active)
    templates.eastern.remap_groups(active,source)
    if category:active['gs_category']=category
    elif 'gs_category' in active:del active['gs_category']
    active['gs_label']=source.get('gs_label','Original '+slot)
    for mod in list(active.modifiers):
        if mod.type=='NODES' and mod.name in ('AR Landmark Fit','AR Eastern Knee Hem'):active.modifiers.remove(mod)
    active.hide_viewport=False;active.hide_render=True
    owner=scene()
    if active.name not in owner.objects:owner.collection.objects.link(active)
    layer=bpy.context.view_layer if bpy.context.scene==owner else owner.view_layers[0]
    layer.update()
    if layer.objects.get(active.name)!=active:
        # Recover this actor alone, without un-excluding an entire user collection.
        if owner.collection.objects.get(active.name)!=active:owner.collection.objects.link(active)
        layer.update()
        print('GARMENT_VISIBILITY: recovered '+active.name+' into Studio root collection.')
    if not visibility.hide_in_layer(active,True,layer):
        raise ValueError('Garment actor is unavailable in the Studio view layer; activation was cancelled safely.')

def activate(identifier):
    s=guard();obj,row=item(identifier);return_to_ar()
    if row.get('profile','MALE')!=body_profiles.current():raise ValueError('Switch to the garment body profile first')
    category=row['category'];slot=templates.CATEGORIES[category][1]
    if category=='FEMALE_DUPATTA':
        import accessories
        accessories.activate(identifier);return
    active=bpy.data.objects[slot]
    previous=(active.data,active.get('gs_category'),active.get('gs_label'),active.get('gs_profile'),active.get('gs_id'))
    try:
        prepare_active(slot,obj,category)
        if slot=='Tops':API.prepare_shirt()
        API.garment_fit.GarmentFitter(bpy.data.objects['Armature'])
    except Exception:
        active.data=previous[0]
        for key,value in zip(['gs_category','gs_label','gs_profile','gs_id'],previous[1:]):
            if value is None:
                if key in active:del active[key]
            else:active[key]=value
        API.garment_fit.GarmentFitter(bpy.data.objects['Armature'])
        raise
    if slot=='Bottoms':s.far_outfit='OUTFIT'
    import eastern_live
    eastern_live.lighting(s)
    body_profiles.apply_regions();body_profiles.capture()
    s.gs_status='Active: '+row['name']+'. Start AR Camera and calibrate for the wearer.'

def restore():
    s=guard();return_to_ar()
    if body_profiles.current()!='MALE':raise ValueError('Original Western outfit belongs to Male; switch to Male first')
    for slot in ['Tops','Bottoms']:prepare_active(slot,bpy.data.objects['GS Original '+slot],None)
    API.prepare_shirt();API.garment_fit.GarmentFitter(bpy.data.objects['Armature'])
    import eastern_live
    eastern_live.lighting(s)
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
    eastern=row['category'] in templates.eastern.CATEGORIES or row['category'] in female_patterns.CATEGORIES
    for name,original in [('GS Preview Key',1000),('GS Preview Fill',700)]:
        lamp=preview_scene.objects.get(name)
        if lamp and lamp.type=='LIGHT':lamp.data.energy=original*(.12 if eastern else 1.)
    old=bpy.data.objects.get('GS Preview Garment')
    if old:bpy.data.objects.remove(old,do_unlink=True)
    view=obj.copy();view.name='GS Preview Garment';view.data=obj.data
    preview_scene.collection.objects.link(view);view.parent=None;view.modifiers.clear();view.matrix_world=API.BASE.copy()
    visibility.show_preview(view,preview_scene)
    previous_rig=bpy.data.objects.get('GS Preview Rig')
    if previous_rig:
        data=previous_rig.data;bpy.data.objects.remove(previous_rig,do_unlink=True)
        if data.users==0:bpy.data.armatures.remove(data)
    if (obj.get('gs_photo_method') or row['category'] in templates.eastern.CATEGORIES or row['category'] in female_patterns.CATEGORIES) and scene().gs_relaxed_preview:
        original=bpy.data.objects['Armature'];rig=original.copy();rig.data=original.data.copy();rig.name='GS Preview Rig'
        rig.animation_data_clear();preview_scene.collection.objects.link(rig);rig.parent=None;rig.matrix_world=API.BASE.copy();rig.data.pose_position='POSE'
        visibility.show_preview(rig,preview_scene)
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

class GS_OT_outfit(bpy.types.Operator):
    bl_idname='garment_studio.outfit';bl_label='Wear complete suit'
    def execute(self,context):
        try:activate_outfit(scene().gs_outfit)
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
        l.prop(s,'gs_profile',expand=True)
        box=l.box();box.label(text=s.gs_profile.title()+' permanent suit collection')
        box.prop(s,'gs_outfit',text='Suit');box.operator('garment_studio.outfit')
        if s.gs_profile=='FEMALE':
            box=l.box();box.label(text='Female visual fit (independent from Male)')
            for key in female_patterns.CONTROLS:box.prop(s,'gs_female_'+key)
            box=l.box();box.label(text='Female Kameez style')
            for key in female_patterns.STYLES:box.prop(s,'gs_female_style_'+key)
        if context.scene.name==PREVIEW:
            row=l.row()
            for x in ['FRONT','BACK','LEFT','RIGHT']:row.operator('garment_studio.angle',text=x.title()).angle=x
            l.operator('garment_studio.return_ar')
            l.operator('garment_studio.activate')
            l.prop(s,'gs_relaxed_preview');l.operator('garment_studio.preview',text='Refresh preview pose')
        box=l.box();box.label(text='1. Front + back garment photos')
        box.prop(s,'gs_name');box.prop(s,'gs_category');box.prop(s,'gs_front');box.prop(s,'gs_back')
        if s.gs_category in ('HALF_SHIRT','FULL_SHIRT') or s.gs_category in templates.eastern.CATEGORIES:box.prop(s,'gs_layout')
        if s.gs_layout=='PRODUCT' and s.gs_category in ('HALF_SHIRT','FULL_SHIRT'):
            box.prop(s,'gs_neckline');box.prop(s,'gs_slim')
            box.label(text='Product photo: centred torso, no row stretching')
        elif s.gs_category not in templates.eastern.CATEGORIES or s.gs_layout=='FLAT':
            box.prop(s,'gs_auto_crop')
            if s.gs_auto_crop:box.prop(s,'gs_threshold')
        box.prop(s,'gs_flip_back')
        if s.gs_layout=='PRODUCT' and (s.gs_category in ('HALF_SHIRT','FULL_SHIRT') or s.gs_category in templates.eastern.CATEGORIES):
            box.label(text='Whole garment visible; plain background')
        else:box.label(text='Flat, straight, sleeves spread; plain background')
        if s.gs_category in templates.eastern.CATEGORIES:
            box.label(text='Eastern v2: reusable 3D master + photo profiles')
            box.label(text='Product layout: sleeves down; matching front/back')
            box.label(text='Long garment: show hips, knees and ankles')
        box.operator('garment_studio.build')
        box=l.box();box.label(text='2. Library - inspect before replacing')
        box.prop(s,'gs_selected',text='Garment');box.operator('garment_studio.preview');box.operator('garment_studio.activate')
        box=l.box();box.label(text='3. Fahad AR tab > Start AR Camera')
        box.label(text='Calibrate for wearer; hips/knees/ankles visible')
        box.label(text='Stop camera before switching garments')
        if s.gs_profile=='MALE':box.operator('garment_studio.restore')
        for slot in ['Tops','Bottoms']:box.label(text=slot+': '+bpy.data.objects[slot].get('gs_label','Original'))
        l.label(text=s.gs_status)
        l.label(text='Template approximation, not exact tailoring')

CLASSES=[GS_OT_build,GS_OT_preview,GS_OT_angle,GS_OT_activate,GS_OT_restore,GS_OT_outfit,GS_OT_return,GS_PT_panel]
def register(ar):
    global API
    API=ar;guard(stopped=False);templates.ensure_templates()
    for cls in CLASSES:
        old=getattr(bpy.types,cls.__name__,None)
        if old is None and issubclass(cls,bpy.types.Operator):
            prefix,op=cls.bl_idname.split('.')
            old=bpy.types.Operator.bl_rna_get_subclass_py(prefix.upper()+'_OT_'+op)
        if old is None and getattr(cls,'is_registered',False):old=cls
        if old:bpy.utils.unregister_class(old)
        bpy.utils.register_class(cls)
    properties={'gs_name':bpy.props.StringProperty(name='Garment name',default='My garment'),
        'gs_category':bpy.props.EnumProperty(name='Template',items=categories),
        'gs_outfit':bpy.props.EnumProperty(name='Complete suit',items=outfit_choices),
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
    body_profiles.register(ar)
