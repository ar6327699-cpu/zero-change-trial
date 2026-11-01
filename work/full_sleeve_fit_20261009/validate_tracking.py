import sys,json,importlib.util,copy
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'FahadAR'),str(ROOT/'FahadLive')]
from female_frame import camera_frame,arm_visible,current_arm_targets,anchor_wrists
from female_coverage import measure
from female_arm_stabilizer import ArmStabilizer
from fahad_backend import Solver
from dataclasses import asdict
checks=[]
def check(value,name):
 assert value,name
 checks.append(name)
def old(name,path):
 spec=importlib.util.spec_from_file_location(name,str(ROOT/'Backups/full_sleeve_fit_20261009'/path));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
prior=old('prior_arm','FahadAR/female_arm_stabilizer.py')
old_solver=old('prior_solver','FahadLive/fahad_backend.py')
frame=np.full((480,640,3),128,np.uint8);frame[:12]=[0,0,255];frame[-12:]=[255,0,0]
wide=camera_frame(frame,True)
check(np.all(wide[0,80:560]==[0,0,255]) and np.all(wide[-1,80:560]==[255,0,0]),'Entire 4:3 sensor including raised-hand margins is retained')
check(wide.shape==(360,640,3) and not wide[:,:80].any() and not wide[:,560:].any(),'Camera pixels use the same fixed 640x360 landmark canvas')
check(np.array_equal(camera_frame(frame,False),cv2.resize(frame[60:420],(640,360))),'Existing other-profile framing is unchanged')
p=np.zeros((33,4));p[:,:2]=.5;p[:,3]=.95
for j,v in {11:(.57,.3),12:(.43,.3),13:(.62,.43),14:(.38,.43),15:(.66,.57),16:(.34,.57),23:(.55,.6),24:(.45,.6),25:(.55,.76),26:(.45,.76),27:(.55,.94),28:(.45,.94)}.items():p[j,:2]=v
w=p.copy();w[:,:3]*=[1.3,1.5,.5]
moving={'valid':True,'stamp':1.,'regions':{str(g):{'valid':True,'quiet':False,'rigid_translation':False} for g in [2,3,4,5]}}
raw=p.copy();raw[[13,15],0]+=.008;raw[[14,16],0]-=.008
out,ids=current_arm_targets(p,raw,moving,1.)
check(set(ids)=={13,14,15,16} and np.array_equal(out[ids,:2],raw[ids,:2]),'Both moving arms meet the current strong observation')
for mode in ['quiet','unknown','stale','weak','implausible','teleport']:
 evidence=copy.deepcopy(moving);observed=raw.copy()
 if mode=='quiet':
  for r in evidence['regions'].values():r['quiet']=True
 if mode=='unknown':evidence['regions']={}
 if mode=='stale':evidence['stamp']=.5
 if mode=='weak':observed[:,3]=.4
 if mode=='implausible':observed[[15,16],1]=.95
 if mode=='teleport':observed[[11,12],0]+=.1
 result,ids=current_arm_targets(p,observed,evidence,1.)
 check(np.array_equal(result,p) and not ids,f'{mode} does not promote current-detector corrections')
for xy,expected in [((.5,.002),True),((.5,-.02),True),((.5,-.08),False),((1.02,.5),True),((1.10,.5),False)]:
 q=p.copy();q[15,:2]=xy
 check(bool(arm_visible(q,15))==expected,f'Bounded wrist visibility at {xy}')
q=p.copy();q[15,3]=.1;check(not arm_visible(q,15),'Missing wrists are not assigned visibility')
for cls in [Solver,old_solver.Solver]:
 solver=cls();solver.female_livefit=True;solver.best_stability=True;solver.full_sleeves=True
 q=p.copy();q[[13,14],1]=.15;q[[15,16],1]=.002
 payload,info=solver.solve({'t':1.,'pose':q.tolist(),'world':w.tolist()},16/9)
 if cls is Solver:new_info=info
 else:old_info=info
check(new_info['left_arm'] and new_info['right_arm'] and not old_info['left_arm'] and not old_info['right_arm'],'Full solver keeps both observed raised arms at the top boundary')
a=Solver();b=old_solver.Solver()
for model in (a,b):model.female_livefit=False;model.male_livefit=True;model.best_stability=True;model.full_sleeves=True
for n in range(60):
 q=p.copy();q[[13,14,15,16],0]+=.01*np.sin(n/7);raw={'t':1+n/7,'pose':q.tolist(),'world':w.tolist()}
 aa,ai=a.solve(raw);bb,bi=b.solve(raw)
 assert asdict(aa)==asdict(bb) and ai==bi
check(True,'Other-profile solver output and diagnostics unchanged over 60 moving frames')
for profile in ['female_quiet']:
 a=ArmStabilizer();b=prior.ArmStabilizer();rng=np.random.default_rng(14)
 for n in range(120):
  q=p.copy();q[[13,14,15,16],:2]+=rng.normal(0,.0006,(4,2));stamp=1+n/7
  evidence={'valid':True,'stamp':stamp,'regions':{str(g):{'valid':True,'quiet':True} for g in [2,3,4,5]}}
  new=a.update(q,w,stamp,evidence,observed=q);before=b.update(q,w,stamp,evidence)
  check(np.array_equal(new[0],before[0]) and np.array_equal(new[1],before[1]),f'{profile} still sample {n}')
# Missing/stale coverage must be empty, not a remembered silhouette.
check(not measure(None,p,1.)['rows'],'No mask produces no coverage evidence')
mask=np.zeros((360,640),np.float32);mask[100:330,275:366]=1
evidence=measure(mask,p,1.);check(len(evidence['rows'])>=5,'Connected visible torso outline is measured')
check(all(270<r[1]<r[2]<370 for r in evidence['rows']),'Outline samples stay on the observed person component')
unseen=p.copy();unseen[[11,12,23,24],3]=0
check(not measure(mask,unseen,1.)['rows'],'Torso loss clears outline evidence')
check(not measure(np.ones((360,640)),p,1.)['rows'],'Background-wide mask is rejected')
# Make a lateral sleeve touch the same mask component: torso rows must not
# stretch out to its extreme edge.
joined=mask.copy();joined[108:143,235:405]=1
q=p.copy();q[13,:2]=[395/640,143/360];q[14,:2]=[245/640,143/360]
e=measure(joined,q,1.)
check(all(r[2]-r[1]<150 for r in e['rows']),'Lateral sleeve corridors cannot inflate the whole trunk')
# Valid hand root refines XY, while maintaining depth and confidence.
hand=np.tile(p[15],(21,1));hand[0,:2]+=[.010,0]
for j in range(1,21):hand[j,:2]=hand[0,:2]+[(j%5-2)*.005,.01+(j//5)*.005]
hand[5,0]=hand[0,0]-.008;hand[17,0]=hand[0,0]+.008
q,ids=anchor_wrists(p,{'left':hand.tolist()});q=np.array(q)
check(ids==[15] and np.allclose(q[15,:2],hand[0,:2]),'Associated visible palm anchors its own cuff wrist')
check(np.array_equal(q[:,2:],p[:,2:]),'Hand anchor never changes depth or confidence')
bad=hand.copy();bad[:,:2]+=.3
q,ids=anchor_wrists(p,{'left':bad.tolist()});check(not ids and np.array_equal(q,p),'Unassociated hand cannot pull the sleeve')
report={'checks_passed':len(checks),'checks':checks,'note':'Controlled observations and masks; no post-change wearer recording.'}
Path(__file__).with_name('tracking_validation.json').write_text(json.dumps(report,indent=2))
print('TRACKING_VALIDATION',json.dumps({'checks_passed':len(checks),'old_raised_arm':[old_info['left_arm'],old_info['right_arm']],'new_raised_arm':[new_info['left_arm'],new_info['right_arm']]}))
