import sys,json,math,time
sys.dont_write_bytecode=True
from pathlib import Path
import bpy,numpy as np
from mathutils import Matrix
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'FahadAR'))
import ar_runtime as runtime
from female_render_lock import RenderLock
packets=json.loads(Path(__file__).with_name('worker_packets.json').read_text())
arm=bpy.data.objects['Armature'];top=bpy.data.objects['Tops']
binding=runtime.load_binding(arm);writer=runtime.BpyArmaturePoseWriter(arm)
fitter=runtime.garment_fit.GarmentFitter(arm);lock=RenderLock();fit=None;rows=[]
for meta in packets:
 start=time.perf_counter();lock.begin(meta,'QA',arm,fitter,fit)
 frame=runtime.decode_pose_frame(json.dumps({'schema_version':1,'seq':meta['seq'],'captured_at':meta['captured_at'],'status':'ok','pose':meta['pose']}))
 runtime.ensure_live_pose(arm)
 writer.apply(runtime.plan_pose_application(frame.pose,runtime.LimbFilter(),binding=binding,apply_orientation_fix=False),insert_keyframes=False)
 lock.restore_pose(arm);source=runtime.outfit_anchors(arm,fast_female=True)
 anchors=np.array([source[i][[0,2]] for i in [11,12,23,24]])
 result=runtime.calibration.locked_fit(anchors,meta['landmarks'],1.)
 if result is None:continue
 fit,_=result;sx,sz,tx,tz=fit
 arm.matrix_world=Matrix.Translation((tx,0,tz))@Matrix.Diagonal((sx,math.sqrt(sx*sz),sz,1.))@runtime.BASE
 info=fitter.update(meta['landmarks'],source,fit,meta['captured_at']);lock.restore_residuals(fitter)
 _,mod,sockets=fitter.items['Tops'];fitter.coverage.update(meta['female_coverage'],meta['captured_at'],meta['landmarks'],sockets,mod,lock.held)
 bpy.context.view_layer.update();obj=top.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=obj.to_mesh()
 coords=np.array([tuple(v.co) for v in mesh.vertices]);assert np.isfinite(coords).all();assert len(mesh.polygons)>0
 obj.to_mesh_clear();rows.append({'seq':meta['seq'],'apply_ms':(time.perf_counter()-start)*1000,'coverage':dict(fitter.coverage.state)})
assert len(rows)>=10 and any(r['coverage'].get('body_rows',0)>=3 for r in rows)
assert fitter.coverage.lining is not None and len(fitter.coverage.lining.data.polygons)>=8
# The source topology still has its authored vents; only the live modifier
# adds the lining, and a reset clears measured shape corrections.
base_faces=len(top.data.polygons);fitter.reset();assert not fitter.coverage.state
report={'packets_applied':len(rows),'native_binding_solver_fit_and_coverage':True,'finite_evaluated_meshes':True,
 'side_lining_faces':len(fitter.coverage.lining.data.polygons),'authored_faces':base_faces,'reset_clears_offsets':True,
 'median_apply_ms':float(np.median([r['apply_ms'] for r in rows])),'rows':rows,'scene_saved':False}
Path(__file__).with_name('runtime_validation.json').write_text(json.dumps(report,indent=2));print('RUNTIME_VALIDATION',json.dumps({k:v for k,v in report.items() if k!='rows'}))
