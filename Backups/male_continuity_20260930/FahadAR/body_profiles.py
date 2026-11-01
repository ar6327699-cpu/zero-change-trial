"""Two isolated fit stores, one runtime skeleton/camera and garment actor pair.

Snapshots store local fit parameters and garment datablocks, never camera frames.
Session calibration objects are deep copied; projection translation is not reused
after a profile switch. Existing Male mesh datablocks are never regionally edited.
"""
import copy
import json
import bpy
from bpy.app.handlers import persistent
import female_patterns as patterns

KEY='GS_body_profiles_v1'
ACTIVE='GS_active_profile'
FIT_PROPS=('far_chest','far_waist','far_depth','far_sleeves','far_neckline','far_shoulders',
           'far_width','far_length','far_x','far_z','far_outfit','far_full_turn','far_occlusion')
LIVE={}
BUSY=False
API=None

def scene():
    import garment_visibility
    return garment_visibility.studio_scene()

def current():return scene().get(ACTIVE,'MALE')

def state(s=None):
    s=s or scene()
    return json.loads(s.get(KEY,'{}'))

def store(data,s=None):(s or scene()).__setitem__(KEY,json.dumps(data))

def slot_name(profile,slot):return 'GS Profile '+profile+' '+slot

def collection():
    s=scene();col=bpy.data.collections.get('GS Body Profiles')
    if col is None:col=bpy.data.collections.new('GS Body Profiles')
    if col.name not in s.collection.children:s.collection.children.link(col)
    col.hide_render=True;col.hide_viewport=True
    return col

def snapshot_slots(profile):
    col=collection()
    for slot in ('Tops','Bottoms'):
        active=bpy.data.objects[slot];name=slot_name(profile,slot);backing=bpy.data.objects.get(name)
        if backing is None:
            backing=active.copy();backing.name=name;col.objects.link(backing)
        # No mesh copy every switch: each profile owns its own retained data.
        backing.data=active.data
        for mod in active.modifiers:
            if mod.type=='ARMATURE':
                saved=next((m for m in backing.modifiers if m.type=='ARMATURE'),None)
                if saved:saved.use_deform_preserve_volume=mod.use_deform_preserve_volume
        backing['gs_profile']=profile
        for key in ('gs_category','gs_label','gs_id','gs_clean_reference_version'):
            if key in active:backing[key]=active[key]
            elif key in backing:del backing[key]
        backing.hide_viewport=True;backing.hide_render=True

def calibration_state(session):
    c=session.calibration
    return {k:copy.deepcopy(getattr(c,k)) for k in ('stage','ratio','ratios','side_counts','back_count','lower_ready','full_outfit','full_turn','message')}

def capture(session=None):
    s=scene();profile=current();data=state(s)
    entry=data.setdefault(profile,{})
    entry['fit']={k:getattr(s,k) for k in FIT_PROPS if hasattr(s,k)}
    entry['regions']={k:getattr(s,'gs_female_'+k,0.) for k in patterns.CONTROLS} if profile=='FEMALE' else {}
    if profile=='FEMALE':entry['style']={k:getattr(s,'gs_female_style_'+k,0.) for k in patterns.STYLES}
    if session is None:session=bpy.app.driver_namespace.get('fahad_ar_runtime')
    if session is not None and getattr(session,'profile_id',None)==profile:
        LIVE[(id(session),profile)]=copy.deepcopy(session.calibration)
        entry['calibration']=calibration_state(session)
    snapshot_slots(profile);store(data,s)

def load_calibration(session):
    import calibration
    profile=current();s=scene();saved=LIVE.get((id(session),profile))
    if saved is not None:c=copy.deepcopy(saved)
    else:
        c=calibration.Calibration(s.far_outfit!='SHIRT',s.far_full_turn)
        saved=state(s).get(profile,{}).get('calibration')
        # Incomplete observation windows cannot survive a restart, but a locked
        # visual fit can. "New Person" explicitly discards only the active one.
        if saved and saved.get('stage') in ('LIVE','REVIEW'):
            for k,v in saved.items():setattr(c,k,copy.deepcopy(v))
    session.calibration=c;session.fit=None;session.profile_id=profile
    s.far_calibration=c.stage

def ensure():
    s=scene();col=collection()
    if ACTIVE not in s:s[ACTIVE]='MALE'
    if KEY not in s:
        male={k:getattr(s,k) for k in FIT_PROPS}
        female=dict(male)
        for k in ('far_chest','far_waist','far_depth','far_sleeves','far_neckline','far_shoulders','far_x','far_z'):female[k]=0.
        female['far_width']=female['far_length']=1.;female['far_outfit']='OUTFIT'
        store({'MALE':{'fit':male,'regions':{}},'FEMALE':{'fit':female,'regions':{k:0. for k in patterns.CONTROLS}}},s)
        snapshot_slots('MALE')
    proxy=patterns.proxy(col)
    # Male proxy is an explicit reference to the existing body, not a replacement.
    male=bpy.data.objects.get('MALE_BODY_PROFILE')
    if male is None:
        male=bpy.data.objects.new('MALE_BODY_PROFILE',None);col.objects.link(male)
        male['gs_existing_body_objects']=json.dumps([o.name for o in s.objects if o.type=='MESH' and o.parent==bpy.data.objects.get('Armature') and o.name not in ('Tops','Bottoms','Shoes') and not o.name.startswith(('GS ','MASTER_','FEMALE_'))])
        male['gs_profile']='MALE';male.hide_viewport=True;male.hide_render=True
    for slot,category in [('Tops','FEMALE_KAMEEZ'),('Bottoms','FEMALE_TROUSER')]:
        name=slot_name('FEMALE',slot)
        if bpy.data.objects.get(name) is None:
            src=patterns.ensure(category,col);obj=src.copy();obj.data=src.data.copy();obj.name=name;col.objects.link(obj)
            obj['gs_label']='Female foundation (select a photo outfit)';patterns.regional_keys(obj)
    proxy['gs_active']=current()=='FEMALE';male['gs_active']=current()=='MALE'
    return proxy

def install_slots(profile):
    # Runtime actor identity is fixed. Its data belongs to only one profile.
    for slot in ('Tops','Bottoms'):
        source=bpy.data.objects[slot_name(profile,slot)];active=bpy.data.objects[slot]
        active.data=source.data
        saved=next((m for m in source.modifiers if m.type=='ARMATURE'),None)
        for mod in active.modifiers:
            if mod.type=='ARMATURE' and saved:mod.use_deform_preserve_volume=saved.use_deform_preserve_volume
        for key in ('gs_category','gs_label','gs_id','gs_profile','gs_clean_reference_version'):
            if key in source:active[key]=source[key]
            elif key in active:del active[key]
        for modifier in list(active.modifiers):
            if modifier.name in ('AR Landmark Fit','AR Eastern Knee Hem','AR Waist Under Shirt'):active.modifiers.remove(modifier)
        active.hide_viewport=False
        import accessories
        accessories.surface(active)
    API.prepare_shirt()
    return API.garment_fit.GarmentFitter(bpy.data.objects['Armature'])

def switch(target):
    global BUSY
    if target not in ('MALE','FEMALE'):raise ValueError('Unknown body profile')
    if BUSY:return
    s=scene();ensure();previous=current()
    if target==previous:return
    session=bpy.app.driver_namespace.get('fahad_ar_runtime');running=session and session.running
    capture(session);data=state(s);BUSY=True
    try:
        API.clothes(False);s[ACTIVE]=target
        for key,value in data[target]['fit'].items():setattr(s,key,value)
        for key,value in data['FEMALE']['regions'].items():setattr(s,'gs_female_'+key,value)
        for key in patterns.STYLES:setattr(s,'gs_female_style_'+key,data['FEMALE'].get('style',{}).get(key,0.))
        fitter=install_slots(target)
        if running:
            session.garments=fitter;load_calibration(session)
            import stability
            session.fit_lock=stability.FitLock();session.last_good_time=None
            session.best_stability=API.stability_enabled()
            session.western_lower.reset()
            if session.conn:
                long=bpy.data.objects['Tops'].get('gs_category') in ('FULL_SHIRT','KURTA','FEMALE_KAMEEZ')
                session.conn.sendall(b'SLEEVE_LONG\n' if long else b'SLEEVE_SHORT\n')
                session.conn.sendall(b'STABILITY_BEST\n' if session.best_stability else b'STABILITY_STANDARD\n')
                session.conn.sendall(('PROFILE_'+API.tracking_profile().upper()+'\n').encode())
        if hasattr(s,'gs_profile'):s.gs_profile=target
        s.gs_category='FEMALE_KAMEEZ' if target=='FEMALE' else 'KURTA'
        if hasattr(s,'gs_outfit'):s.gs_outfit='NONE'
        ensure();apply_regions()
        import accessories
        accessories.sync()
        import eastern_live
        eastern_live.lighting(s)
        s.gs_selected='NONE';s.gs_status=target+' profile active; shared tracking retained.'
        if not running:s.far_calibration='FRONT'
    except Exception:
        s[ACTIVE]=previous
        for key,value in data[previous]['fit'].items():setattr(s,key,value)
        fitter=install_slots(previous)
        if running:session.garments=fitter;load_calibration(session)
        if hasattr(s,'gs_profile'):s.gs_profile=previous
        ensure();raise
    finally:BUSY=False

def apply_regions():
    s=scene();values={k:getattr(s,'gs_female_'+k,0.) for k in patterns.CONTROLS}
    patterns.apply_regions(bpy.data.objects.get('FEMALE_BODY_PROXY'),values)
    if current()=='FEMALE':
        for name in ('Tops','Bottoms'):patterns.apply_regions(bpy.data.objects.get(name),values)
        top=bpy.data.objects.get('Tops')
        if top and top.data.shape_keys:
            for name in patterns.STYLES:
                key=top.data.shape_keys.key_blocks.get('Style '+name)
                if key:key.value=getattr(s,'gs_female_style_'+name,0.)

def region_changed(self,context):
    if not BUSY:apply_regions()

def profile_changed(self,context):
    if BUSY:return
    try:switch(self.gs_profile)
    except Exception as e:self.gs_status='Profile switch failed: '+str(e)

@persistent
def save_profiles(_):
    if API is not None and any(s.get('GS_studio_version') for s in bpy.data.scenes):capture()

def register(api):
    global API,BUSY
    API=api;BUSY=True
    try:
        bpy.types.Scene.gs_profile=bpy.props.EnumProperty(name='Body / Profile',items=[('MALE','Male','Existing Male Western + Eastern'),('FEMALE','Female','Independent Female Eastern fit')],update=profile_changed)
        for key in patterns.CONTROLS:
            setattr(bpy.types.Scene,'gs_female_'+key,bpy.props.FloatProperty(name=key.replace('_',' ').title(),min=-1,max=1,default=0,update=region_changed))
        for key in patterns.STYLES:
            setattr(bpy.types.Scene,'gs_female_style_'+key,bpy.props.FloatProperty(name=key.replace('_',' ').title(),min=-1,max=1,default=0,update=region_changed))
        ensure();s=scene();s.gs_profile=current()
        for key,value in state(s)['FEMALE']['regions'].items():setattr(s,'gs_female_'+key,value)
        for key,value in state(s)['FEMALE'].get('style',{}).items():setattr(s,'gs_female_style_'+key,value)
        apply_regions()
        for handler in list(bpy.app.handlers.save_pre):
            if getattr(handler,'__module__','')==__name__:bpy.app.handlers.save_pre.remove(handler)
        bpy.app.handlers.save_pre.append(save_profiles)
    finally:BUSY=False
