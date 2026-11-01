"""Blender runtime for the AR COPY ONLY. Run this text once, then N > Fahad AR.

The original .blend, its meshes and the installed PoseCap addon are not edited.
"""
import bpy,json,math,socket,struct,threading,queue,subprocess,time,os,sys,importlib
from pathlib import Path
import numpy as np
from mathutils import Matrix,Vector
from posecap_contracts import decode_pose_frame
from posecap_core import plan_pose_application,LimbFilter
from bl_ext.user_default.posecap.posecap_addon.binding_state import load_binding
from bl_ext.user_default.posecap.posecap_addon.apply_timer import BpyArmaturePoseWriter

KEY='fahad_ar_runtime';W=640;H=360;CW=32/9;CH=2.
BASE=Matrix.Rotation(math.pi/2,4,'X')@Matrix.Diagonal((.01,.01,.01,1.))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import calibration
importlib.reload(calibration)
import garment_fit
importlib.reload(garment_fit)

def prepare_shirt():
    """Non-destructive shape keys on the AR shirt only; rest UVs unchanged.

    Dimensions here are visual trims, not inferred circumferences. Weights
    isolate torso and sleeve regions with smooth transitions to avoid seams.
    """
    assert_copy()
    obj=bpy.data.objects['Tops']
    if not obj.data.shape_keys:obj.shape_key_add(name='Basis')
    basis=obj.data.shape_keys.key_blocks[0]
    coords=np.array([v.co[:] for v in basis.data]);y=coords[:,1];x=coords[:,0]
    arm=bpy.data.objects['Armature']
    shoulder=arm.data.bones['mixamorig:LeftArm'].head_local.x
    sy=arm.data.bones['mixamorig:LeftArm'].head_local.y
    hip=arm.data.bones['mixamorig:Hips'].head_local.y
    torso=np.clip((shoulder*1.2-np.abs(x))/(shoulder*.45),0,1)
    torso=torso*torso*(3-2*torso)
    factors={'Chest':torso*np.exp(-((y-(sy-25))/30)**2),
             'Waist':torso*np.exp(-((y-(hip+22))/33)**2),
             'Depth':np.ones(len(coords)),
             'Sleeves':1-torso}
    for name,weight in factors.items():
        key=obj.data.shape_keys.key_blocks.get('AR '+name)
        if key is None:key=obj.shape_key_add(name='AR '+name)
        dest=coords.copy()
        if name=='Depth':dest[:,2]*=1.3
        elif name=='Sleeves':
            dest[:,0]+=np.sign(x)*np.maximum(np.abs(x)-shoulder,0)*.3*weight
        else:
            dest[:,0]*=1+.25*weight
            dest[:,2]*=1+.15*weight
        key.data.foreach_set('co',dest.ravel());key.slider_min=-1;key.slider_max=1
    neck=obj.data.shape_keys.key_blocks.get('AR Neckline') or obj.shape_key_add(name='AR Neckline')
    dest=coords.copy()
    central=np.exp(-(x/(shoulder*.65))**4)
    upper=np.clip((y-(sy-18))/30,0,1);upper=upper*upper*(3-2*upper)
    dest[:,1]-=18*central*upper
    neck.data.foreach_set('co',dest.ravel());neck.slider_min=0;neck.slider_max=1
    cap=obj.data.shape_keys.key_blocks.get('AR Shoulder caps') or obj.shape_key_add(name='AR Shoulder caps')
    dest=coords.copy()
    lateral=np.clip((np.abs(x)-shoulder*.32)/(shoulder*.50),0,1)
    lateral=lateral*lateral*(3-2*lateral)
    # Lower the excess rest-mesh cap height, not the entire shirt or skeleton.
    excess=3*np.logaddexp(0,(y-(sy-1))/3)
    dest[:,1]-=excess*lateral
    cap.data.foreach_set('co',dest.ravel());cap.slider_min=0;cap.slider_max=1
    update_shirt()

def update_shirt():
    obj=bpy.data.objects.get('Tops');s=bpy.context.scene
    if not obj or not obj.data.shape_keys:return
    for name,prop in [('Chest','far_chest'),('Waist','far_waist'),('Depth','far_depth'),('Sleeves','far_sleeves'),('Neckline','far_neckline'),('Shoulder caps','far_shoulders')]:
        key=obj.data.shape_keys.key_blocks.get('AR '+name)
        value=getattr(s,prop,0.)
        if key and abs(key.value-value)>1e-6:key.value=value

def prepare_foreground():
    """Real RGB + semantic alpha in front of garment, in camera coordinates."""
    assert_copy()
    matte=bpy.data.images.get('AR Foreground Matte')
    if matte is None:matte=bpy.data.images.new('AR Foreground Matte',width=W,height=H,alpha=False)
    matte.colorspace_settings.name='Non-Color'
    plane=bpy.data.objects.get('AR Real Foreground')
    if plane is None:
        mesh=bpy.data.meshes.new('AR Foreground Plane')
        mesh.from_pydata([(-CW/2,-5,-1),(CW/2,-5,-1),(CW/2,-5,1),(-CW/2,-5,1)],[],[(0,1,2,3)])
        uv=mesh.uv_layers.new(name='UVMap')
        for loop,co in zip(uv.data,[(0,0),(1,0),(1,1),(0,1)]):loop.uv=co
        plane=bpy.data.objects.new('AR Real Foreground',mesh);bpy.context.scene.collection.objects.link(plane)
        mat=bpy.data.materials.new('AR Foreground Composite');mat.use_nodes=True
        mat.surface_render_method='BLENDED'
        nodes=mat.node_tree.nodes;nodes.clear();links=mat.node_tree.links
        rgb=nodes.new('ShaderNodeTexImage');rgb.image=bpy.data.images['AR Live Camera']
        alpha=nodes.new('ShaderNodeTexImage');alpha.image=matte
        em=nodes.new('ShaderNodeEmission');transparent=nodes.new('ShaderNodeBsdfTransparent')
        mix=nodes.new('ShaderNodeMixShader');out=nodes.new('ShaderNodeOutputMaterial')
        links.new(rgb.outputs['Color'],em.inputs['Color']);links.new(alpha.outputs['Color'],mix.inputs[0])
        links.new(transparent.outputs[0],mix.inputs[1]);links.new(em.outputs[0],mix.inputs[2]);links.new(mix.outputs[0],out.inputs['Surface'])
        mesh.materials.append(mat)
    plane.hide_set(False);plane.hide_render=False
    set_matte(None)

def set_matte(data):
    image=bpy.data.images.get('AR Foreground Matte')
    if image is None:return
    pixels=np.ones((H,W,4),np.float32)
    if data is None:pixels[:,:,:3]=0
    else:pixels[:,:,:3]=np.frombuffer(data,np.uint8).reshape(H,W)[::-1,:,None]/255.
    image.pixels.foreach_set(pixels.ravel());image.update()

def anchors(arm):
    bpy.context.view_layer.update()
    return np.array([tuple(BASE@arm.pose.bones['mixamorig:'+n].head) for n in
         ['LeftArm','RightArm','LeftUpLeg','RightUpLeg']])[:,[0,2]]

def outfit_anchors(arm):
    bpy.context.view_layer.update()
    return {i:np.array(BASE@arm.pose.bones['mixamorig:'+n].head) for i,n in garment_fit.CONTROL_BONES.items()}

def fitted_projection(arm,p,profile,previous,src=None):
    s=bpy.context.scene
    # Width calibration is applied in projected space; profile depth stays an
    # explicit mesh-local visual control, never inferred from loose clothing.
    result=calibration.locked_fit(anchors(arm) if src is None else src,p,profile.ratio,previous,
        s.far_width,s.far_length,s.far_x,s.far_z)
    if result is None:return None
    target,error=result;sx,sz,tx,tz=target
    arm.matrix_world=Matrix.Translation((tx,0,tz))@Matrix.Diagonal((sx,math.sqrt(sx*sz),sz,1.))@BASE
    return target,error

def assert_copy():
    if not bpy.context.scene.get('Fahad_AR_copy') or Path(bpy.data.filepath).name!='fahadGenius_AR_experiment.blend':
        raise RuntimeError('AR is restricted to fahadGenius_AR_experiment.blend. Original avatar is protected.')

def clothes(visible=True,p=None,lower_ready=True):
    s=bpy.context.scene
    for name in ['Tops','Bottoms','Shoes']:
        obj=bpy.data.objects.get(name)
        if obj:
            show=visible and (name=='Tops' or (name=='Bottoms' and s.far_outfit in ('OUTFIT','FULL')) or s.far_outfit=='FULL')
            if name!='Tops' and visible and p is not None:
                valid=calibration.observed(p,.5)
                pairs=[(23,25),(24,26)] if name=='Bottoms' else [(25,27),(26,28)]
                show=show and lower_ready and valid is not None and any(valid[a] and valid[b] for a,b in pairs)
                if name=='Bottoms' and obj.get('gs_category') in ('PANTS','TROUSERS'):
                    show=show and valid is not None and any(all(valid[i] for i in side) for side in [(23,25,27),(24,26,28)])
            if obj.hide_get()==show:obj.hide_set(not show)
            if obj.hide_render==show:obj.hide_render=not show

def placeholder():
    set_matte(None)
    image=bpy.data.images.get('AR Live Camera')
    if image:
        a=np.zeros((H,W,4),dtype=np.float32);a[:,:,:3]=.055;a[:,:,3]=1
        image.pixels.foreach_set(a.ravel());image.update()

def fit_armature(arm,landmarks,previous=None):
    """Fit projected shoulder width + torso height, independently of depth.

    This is front-view overlay alignment, NOT physical garment size inference.
    """
    p=np.asarray(landmarks,dtype=float)
    if p.shape!=(33,4) or not np.isfinite(p).all():return None
    ids=[11,12,23,24]
    if any(p[i,3]<.5 or not (.015<p[i,0]<.985 and .015<p[i,1]<.985) for i in ids):return None
    if abs(p[11,0]-p[12,0])<.035:return None
    names=['LeftArm','RightArm','LeftUpLeg','RightUpLeg']
    arm.matrix_world=BASE.copy();bpy.context.view_layer.update()
    src=np.array([tuple(arm.matrix_world@arm.pose.bones['mixamorig:'+n].head) for n in names])[:,[0,2]]
    dst=np.c_[(p[ids,0]-.5)*CW,(.5-p[ids,1])*CH]
    width=src[0,0]-src[1,0];height=(src[0,1]+src[1,1]-src[2,1]-src[3,1])*.5
    if abs(width)<.05 or height<.05:return None
    sx=(dst[0,0]-dst[1,0])/width
    sz=((dst[0,1]+dst[1,1]-dst[2,1]-dst[3,1])*.5)/height
    if not (.07<sx<3 and .07<sz<3):return None
    scale=np.array([sx,sz]);offset=np.mean(dst-src*scale,axis=0)
    # Smooth placement once, lightly; both image and pose are still one frame.
    target=np.r_[scale,offset]
    if previous is not None:target=previous*.25+target*.75
    sx,sz,tx,tz=target
    scene=bpy.context.scene
    sx*=scene.far_width;sz*=scene.far_length
    # Apply manual size adjustment about the measured shoulder centre.
    anchor=src[:2].mean(axis=0)
    tx+=(target[0]-sx)*anchor[0]+scene.far_x
    tz+=(target[1]-sz)*anchor[1]+scene.far_z
    arm.matrix_world=Matrix.Translation((tx,0,tz))@Matrix.Diagonal((sx,math.sqrt(sx*sz),sz,1.))@BASE
    bpy.context.view_layer.update()
    actual=np.array([tuple(arm.matrix_world@arm.pose.bones['mixamorig:'+n].head) for n in names])[:,[0,2]]
    error=float(np.sqrt(np.mean(((actual-dst)*np.array([W/CW,H/CH]))**2)))
    return target,error

class Session:
    def __init__(self):
        self.q=queue.Queue(maxsize=1);self.running=True;self.conn=None;self.proc=None
        self.listener=socket.socket();self.listener.bind(('127.0.0.1',0));self.listener.listen(1);self.listener.settimeout(1)
        self.arm=bpy.data.objects['Armature'];self.binding=load_binding(self.arm);self.writer=BpyArmaturePoseWriter(self.arm)
        self.fit=None;self.frames=0;self.last=time.monotonic();self.started=self.last;self.errors=[];self.latencies=[]
        s=bpy.context.scene
        self.log=None;self.calibration=calibration.Calibration(s.far_outfit!='SHIRT',s.far_full_turn);self.valid_frames=0;self.occlusion_frames=0
        self.garments=garment_fit.GarmentFitter(self.arm);self.region_info={}
        self.inference_times=[];self.apply_times=[];self.mode=bpy.context.scene.far_quality
    def put(self,value):
        try:self.q.put_nowait(value)
        except queue.Full:
            try:self.q.get_nowait()
            except queue.Empty:pass
            self.q.put_nowait(value)
    def read(self):
        try:
            while self.running:
                try:self.conn,_=self.listener.accept();break
                except socket.timeout:continue
            if not self.running:return
            self.conn.settimeout(2)
            def receive(n):
                data=bytearray()
                while len(data)<n and self.running:
                    try:part=self.conn.recv(n-len(data))
                    except socket.timeout:continue
                    if not part:raise EOFError('Camera connection closed')
                    data.extend(part)
                if len(data)!=n:raise EOFError('Stopped')
                return bytes(data)
            while self.running:
                a,b=struct.unpack('!II',receive(8))
                if a>1000000 or b not in (0,W*H*3,W*H*4):raise ValueError('Invalid local camera packet')
                self.put((json.loads(receive(a)),receive(b)))
        except Exception as e:
            if self.running:self.put(({'kind':'error','message':str(e)},b''))
    def start(self):
        folder=Path(bpy.data.filepath).parent/'FahadAR'
        # Resolve within the interactive process's own LocalAppData view.
        registry=Path(os.environ['LOCALAPPDATA'])/'PoseCap/backends/mediapipe'
        python=registry/'runtime/Scripts/python.exe';model=registry/'models/holistic_landmarker.task'
        if not python.is_file() or not model.is_file():raise RuntimeError('Existing MediaPipe runtime/model not found.')
        threading.Thread(target=self.read,daemon=True).start()
        self.log=open(folder/'runtime.log','a',encoding='utf-8')
        self.proc=subprocess.Popen([str(python),str(folder/'ar_camera.py'),'--port',str(self.listener.getsockname()[1]),
            '--model',str(model),'--camera',str(bpy.context.scene.far_camera),'--quality',self.mode,
            '--sleeves','long' if bpy.data.objects['Tops'].get('gs_category')=='FULL_SHIRT' else 'short'],stdout=self.log,stderr=self.log,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        bpy.app.timers.register(self.tick,first_interval=.05)
    def tick(self):
        if not self.running:return None
        if not bpy.context.scene.get('Fahad_AR_copy'):
            self.stop();return None
        try:meta,rgb=self.q.get_nowait()
        except queue.Empty:
            if time.monotonic()-self.last>2:clothes(False)
            if self.proc and self.proc.poll() is not None:
                bpy.context.scene.far_status='Camera stopped. Check runtime.log; Start AR to retry.';self.stop();return None
            if time.monotonic()-self.started>90 and not self.frames:
                bpy.context.scene.far_status='Camera startup timed out. Check camera index/log.';self.stop();return None
            return .005
        try:
            s=bpy.context.scene
            apply_started=time.perf_counter()
            if self.calibration.full_outfit!=(s.far_outfit!='SHIRT') or self.calibration.full_turn!=s.far_full_turn:
                self.calibration=calibration.Calibration(s.far_outfit!='SHIRT',s.far_full_turn)
                self.fit=None;self.garments.reset();clothes(False)
                if self.conn:self.conn.sendall(b'CAL\n')
            if meta['kind']!='frame':
                s.far_status=meta.get('message','Camera status')
                if meta['kind']=='error':self.stop();return None
                return .005
            self.last=time.monotonic();self.frames+=1
            pixels=np.ones((H,W,4),dtype=np.float32)
            pixels[:,:,:3]=np.frombuffer(rgb[:W*H*3],np.uint8).reshape(H,W,3)[::-1]/255.
            image=bpy.data.images['AR Live Camera'];image.pixels.foreach_set(pixels.ravel());image.update()
            matte=rgb[W*H*3:] if len(rgb)==W*H*4 else None
            set_matte(matte if s.far_occlusion else None)
            if matte and meta.get('occlusion',{}).get('arms',0):self.occlusion_frames+=1
            frame=decode_pose_frame(json.dumps({'schema_version':1,'seq':meta['seq'],'captured_at':meta['captured_at'],'status':'ok','pose':meta['pose']}))
            ensure_live_pose(self.arm)
            self.writer.apply(plan_pose_application(frame.pose,LimbFilter(),binding=self.binding,apply_orientation_fix=False),insert_keyframes=False)
            control_source=outfit_anchors(self.arm)
            src=np.array([control_source[i][[0,2]] for i in [11,12,23,24]])
            self.calibration.update(meta['landmarks'],src,meta.get('info',{}).get('heading_deg',0),
                meta['captured_at'],meta.get('world'))
            s.far_calibration=self.calibration.stage
            update_shirt()
            result=fitted_projection(self.arm,meta['landmarks'],self.calibration,self.fit,src) if self.calibration.ratio else None
            if self.calibration.ratio is None:
                clothes(False);s.far_status=self.calibration.message
            elif result is None:
                clothes(False);s.far_status='Tracking lost: show at least one shoulder + same-side hip.'
            else:
                self.fit,error=result;self.errors.append(error);self.valid_frames+=1
                self.region_info=self.garments.update(meta['landmarks'],control_source,self.fit)
                clothes(True,meta['landmarks'],self.calibration.lower_ready)
                s.far_lower_status=('Shorts tracking' if not bpy.data.objects['Bottoms'].hide_get() else 'Show hips + knees for shorts') if s.far_outfit!='SHIRT' else 'Shirt-only mode'
                elapsed=time.time()-meta['captured_at']
                s.far_status=(f'LIVE | {error:.0f}px joints | {elapsed*1000:.0f}ms | real arms {meta.get("occlusion",{}).get("arms",0)}'
                    if self.calibration.stage=='LIVE' else self.calibration.message)
            for screen in bpy.data.screens:
                for area in screen.areas:
                    if area.type=='VIEW_3D':area.tag_redraw()
            self.latencies.append(time.time()-meta['captured_at'])
            self.inference_times.append(meta.get('inference_ms',0));self.apply_times.append((time.perf_counter()-apply_started)*1000)
            # Numerical heartbeat only, no personal profile or camera images.
            if self.frames%30==0:
                report={'mode':self.mode,'frames':self.frames,'fit_frames':self.valid_frames,
                    'capture_to_apply_ms':float(np.median(self.latencies[-60:])*1000) if self.latencies else None,
                    'inference_ms':float(np.median(self.inference_times[-60:])),
                    'blender_apply_ms':float(np.median(self.apply_times[-60:])),
                    'processed_fps':self.frames/max(time.monotonic()-self.started,.01),'stage':self.calibration.stage,
                    'lower_calibrated':self.calibration.lower_ready,'back_samples':self.calibration.back_count,
                    'regional_fit':self.region_info}
                (Path(bpy.data.filepath).parent/'FahadAR/performance_live.json').write_text(json.dumps(report,indent=2))
        except Exception as e:
            bpy.context.scene.far_status='AR error: '+str(e);self.stop();return None
        return .005
    def stop(self):
        if not self.running:return
        self.running=False
        if self.conn:
            try:self.conn.sendall(b'STOP\n');self.conn.shutdown(socket.SHUT_RDWR)
            except OSError:pass
            self.conn.close()
        self.listener.close()
        if self.proc and self.proc.poll() is None:
            try:self.proc.wait(timeout=.5)
            except subprocess.TimeoutExpired:self.proc.terminate()
        if self.log:self.log.close()
        if bpy.context.scene.get('Fahad_AR_copy'):
            clothes(False);placeholder()
            report={'frames_applied':self.frames,'valid_fit_frames':self.valid_frames,'calibration_stage':self.calibration.stage,
                'mode':self.mode,'inference_median_ms':float(np.median(self.inference_times)) if self.inference_times else None,
                'blender_apply_median_ms':float(np.median(self.apply_times)) if self.apply_times else None,
                'side_sample_counts':self.calibration.side_counts,'back_samples':self.calibration.back_count,
                'lower_calibrated':self.calibration.lower_ready,'frames_with_foreground_arms':self.occlusion_frames,
                'alignment_median_px':float(np.median(self.errors)) if self.errors else None,
                'capture_to_apply_median_ms':float(np.median(self.latencies)*1000) if self.latencies else None,
                'note':'Projection alignment only; not physical fit or full display latency. No camera images saved.'}
            (Path(bpy.data.filepath).parent/'FahadAR/last_session.json').write_text(json.dumps(report,indent=2))

def ensure_live_pose(arm):
    """Restore skinning during AR, without changing stopped inspection mode.

    Rest Position bypasses garment deformation even when tracking writes
    correct bone rotations. Never rewrite the mesh or its saved rest pose.
    """
    if arm.data.pose_position != 'POSE':
        arm.data.pose_position = 'POSE'

class FAR_OT_start(bpy.types.Operator):
    bl_idname='fahad_ar.start';bl_label='Start AR Camera'
    def execute(self,context):
        session=None
        try:
            assert_copy();old=bpy.app.driver_namespace.get(KEY)
            if old and old.running:old.stop()
            if context.scene.posecap.lifecycle_state!='STOPPED':bpy.ops.posecap.stop_stream()
            ensure_live_pose(bpy.data.objects['Armature'])
            for bone in bpy.data.objects['Armature'].pose.bones:bone.matrix_basis=Matrix.Identity(4)
            clothes(False);session=Session();bpy.app.driver_namespace[KEY]=session
            context.scene.far_status='Starting local camera...';session.start()
        except Exception as e:
            if session:session.stop()
            context.scene.far_status='Start failed: '+str(e)
            self.report({'ERROR'},str(e));return {'CANCELLED'}
        return {'FINISHED'}
class FAR_OT_stop(bpy.types.Operator):
    bl_idname='fahad_ar.stop';bl_label='Stop AR Camera'
    def execute(self,context):
        session=bpy.app.driver_namespace.get(KEY)
        if session:session.stop()
        context.scene.far_status='Stopped. Camera image cleared; original file unchanged.'
        return {'FINISHED'}
class FAR_OT_calibrate(bpy.types.Operator):
    bl_idname='fahad_ar.calibrate';bl_label='New Person / Recalibrate'
    def execute(self,context):
        session=bpy.app.driver_namespace.get(KEY)
        if session and session.running and session.conn:
            session.conn.sendall(b'CAL\n');session.fit=None
            session.calibration=calibration.Calibration(context.scene.far_outfit!='SHIRT',context.scene.far_full_turn)
            session.garments.reset()
            context.scene.far_calibration='FRONT';clothes(False)
        return {'FINISHED'}
class FAR_OT_confirm(bpy.types.Operator):
    bl_idname='fahad_ar.confirm';bl_label='Confirm Visual Fit'
    def execute(self,context):
        session=bpy.app.driver_namespace.get(KEY)
        if not session or not session.running or not session.calibration.confirm():
            self.report({'WARNING'},'Complete front and both side samples first.');return {'CANCELLED'}
        context.scene.far_calibration='LIVE';return {'FINISHED'}
class FAR_PT_panel(bpy.types.Panel):
    bl_label='Fahad AR - COPY ONLY';bl_idname='FAR_PT_panel';bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='Fahad AR'
    @classmethod
    def poll(cls,context):return context.scene.get('Fahad_AR_copy',False)
    def draw(self,context):
        l=self.layout;s=context.scene;l.label(text='Real camera + existing clothes')
        l.label(text='Guided visual fit | not a body scanner')
        l.prop(s,'far_camera');l.prop(s,'far_outfit');l.prop(s,'far_quality');l.prop(s,'far_full_turn')
        l.operator('fahad_ar.start');l.operator('fahad_ar.stop');l.operator('fahad_ar.calibrate')
        l.label(text='Stage: '+s.far_calibration)
        l.prop(s,'far_occlusion');l.prop(s,'far_neckline');l.prop(s,'far_shoulders')
        if s.far_calibration=='REVIEW':l.operator('fahad_ar.confirm')
        l.label(text=s.far_status[:55]);l.label(text=s.far_status[55:110])
        l.label(text=s.far_lower_status)
        l.separator();l.label(text='Loose/open clothes: hidden size unknown')
        l.label(text='Visual trim (not centimetres)')
        for n in ['far_chest','far_waist','far_depth','far_sleeves']:l.prop(s,n)
        l.separator();l.label(text='Overall alignment trim')
        for n in ['far_width','far_length','far_x','far_z']:l.prop(s,n)
        l.label(text='Show shoulders, hips AND knees; one person.')
        l.label(text='Camera view is not mirrored.')

def register():
    assert_copy()
    bpy.context.scene.render.use_compositing=False
    if bpy.context.scene.camera:bpy.context.scene.camera.data.passepartout_alpha=1.0
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':area.spaces.active.shading.use_compositor='DISABLED'
    old=bpy.app.driver_namespace.get(KEY)
    if old and getattr(old,'running',False):old.stop()
    for cls in [FAR_OT_start,FAR_OT_stop,FAR_OT_calibrate,FAR_OT_confirm,FAR_PT_panel]:
        existing=getattr(bpy.types,cls.__name__,None)
        if existing:bpy.utils.unregister_class(existing)
        bpy.utils.register_class(cls)
    props={'far_camera':bpy.props.IntProperty(name='Camera',default=0,min=0,max=8),
       'far_outfit':bpy.props.EnumProperty(name='Clothes',items=[('SHIRT','Existing shirt','Recalibrate after changing clothes mode'),('FULL','Shirt + shorts + shoes','Whole body needed; recalibrate after changing mode'),('OUTFIT','Existing shirt + shorts','Shoulders to knees needed; recalibrate after changing mode')],default='OUTFIT'),
       'far_full_turn':bpy.props.BoolProperty(name='Calibrate back view (restart)',default=True),
       'far_lower_status':bpy.props.StringProperty(default='Show hips and knees for shorts calibration'),
       'far_status':bpy.props.StringProperty(default='Ready. Start AR Camera.'),
       'far_calibration':bpy.props.StringProperty(default='FRONT'),
       'far_quality':bpy.props.EnumProperty(name='Mode (restart)',items=[('smooth','Smooth / adaptive matte','Lower latency; colour-based skin matte can be less accurate'),('semantic','Semantic matte / slower','Neural skin segmentation; greater CPU cost')],default='smooth'),
       'far_chest':bpy.props.FloatProperty(name='Chest',default=0,min=-1,max=1),
       'far_waist':bpy.props.FloatProperty(name='Waist',default=0,min=-1,max=1),
       'far_depth':bpy.props.FloatProperty(name='Depth (side fit)',default=0,min=-1,max=1),
       'far_sleeves':bpy.props.FloatProperty(name='Sleeve length',default=0,min=-1,max=1),
       'far_neckline':bpy.props.FloatProperty(name='Lower neckline',default=0,min=0,max=1),
       'far_occlusion':bpy.props.BoolProperty(name='Real hands / neck in front',default=True),
       'far_shoulders':bpy.props.FloatProperty(name='Relax shoulder caps',default=.25,min=0,max=1),
       'far_width':bpy.props.FloatProperty(name='Width',default=1,min=.7,max=1.4),
       'far_length':bpy.props.FloatProperty(name='Length',default=1,min=.7,max=1.4),
       'far_x':bpy.props.FloatProperty(name='Left / right',default=0,min=-.5,max=.5),
       'far_z':bpy.props.FloatProperty(name='Up / down',default=0,min=-.5,max=.5)}
    for name,prop in props.items():setattr(bpy.types.Scene,name,prop)
    s=bpy.context.scene
    if s.get('AR_fit_version',0)<2:
        s.far_outfit='OUTFIT';s.far_shoulders=.25;s.far_neckline=0;s.far_full_turn=True
        s['AR_fit_version']=2
    prepare_shirt()
    garment_fit.GarmentFitter(bpy.data.objects['Armature'])
    prepare_foreground()
    print('Fahad AR registered. Original avatar protected. N panel > Fahad AR > Start AR Camera.')

if __name__=='__main__':register()
