"""Offline regression checks; no camera, Blender save, or outside-folder writes."""
import copy
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'FahadAR'))
from female_arm_stabilizer import ArmStabilizer, image_still
from female_arm_image import ArmImageFollower
from female_arm_observations import ArmTranslationRepair
from female_reliability import RegionalLock, StillnessGate
from stability import LandmarkLock, QuietRegionLock
from female_stability import SleeveLock

BACKUP=ROOT/'Backups/sleeve_precision_20261005'
def old_module(name):
    spec=importlib.util.spec_from_file_location('old_'+name,BACKUP/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
OLD=old_module('female_arm_stabilizer')
OLD_IMAGE=old_module('female_arm_image')
checks=[]
def check(name,condition):
    assert condition,name
    checks.append(name)

def pose():
    p=np.tile([.5,.5,0.,.99],(33,1))
    for i,xy in {11:(.58,.30),12:(.42,.30),13:(.69,.35),14:(.31,.35),15:(.80,.40),16:(.20,.40),23:(.55,.58),24:(.45,.58),25:(.55,.74),26:(.45,.74),27:(.55,.92),28:(.45,.92)}.items():p[i,:2]=xy
    w=p.copy();w[:,:3]=(p[:,:3]-[.5,.58,0])*[2,2,1]
    return p,w

def evidence(stamp,active=(),supported=None,quiet=True):
    m=dict(valid=True,stamp=stamp,epoch=1,mode='quiet' if quiet else 'moving',quiet=quiet,
           regions={str(g):dict(valid=True,quiet=quiet,rigid_translation=quiet,
           position_epoch=1,position_px=[0.,0.]) for g in range(10)},translation_px=[0.,0.])
    if active:
        f=ArmImageFollower();f.moving=list(active)
        f.state={s:dict(supported_segments=(supported or {'left':[2,3],'right':[4,5]})[s]) for s in active}
        m=f.motion_evidence(m)
    return m

# Exactly the old quiet/unknown behavior, including independent-arm noise.
rng=np.random.default_rng(120502)
for mode in ['quiet','unknown','stale','translation']:
    old=OLD.ArmStabilizer();new=ArmStabilizer()
    for k in range(180):
        p,w=pose();p[[13,15],:2]+=rng.normal(0,.001,(2,2));w[[13,15],:3]+=rng.normal(0,.009,(2,3))
        if mode=='translation':p[:,:2]+=[.02*np.sin(k/30),.01*np.cos(k/30)];w[:,:3]+=[.04*np.sin(k/30),0,0]
        stamp=k/7;m=evidence(stamp)
        if mode=='unknown':m=None
        if mode=='stale':m=evidence(stamp-1,active=['left'])
        a,b=old.update(p,w,stamp,m);c,d=new.update(p,w,stamp,m)
        assert np.array_equal(a,c) and np.array_equal(b,d),(mode,k)
    check(mode+' output bit-for-bit unchanged',True)

# Current-image motion must beat cumulative rigid votes without modifying input.
m=evidence(1);saved=copy.deepcopy(m);f=ArmImageFollower();f.moving=['left'];f.state={'left':{'supported_segments':[2]}}
v=f.motion_evidence(m)
check('motion evidence is non-mutating',m==saved)
check('partial support does not authorize wrist',v['arm_follow_joints']=={'left':[13]})
check('moving arm rejects contradictory rigid vote',image_still(v,1,'left') is None and not v['regions']['2']['rigid_translation'])
check('opposite quiet arm preserved',v['regions']['4']==m['regions']['4'] and image_still(v,1,'right'))

# At low and high rates, track supported XY exactly; retain unobserved joints.
metrics=[]
for fps in [6.3,7,15,30,60]:
    old=OLD.ArmStabilizer();new=ArmStabilizer();errors=[[],[]]
    for k in range(100):
        p,w=pose();stamp=k/fps
        if k>=10:
            offset=18*np.sin((k-10)/fps*2)
            p[13,1]+=offset/720;p[15,1]+=offset/360
        m=evidence(stamp,active=['left'] if k>=10 else [])
        # Reproduce old upstream metadata: quiet overridden but rigid not.
        legacy=copy.deepcopy(m)
        if k>=10:
            for g in (2,3):legacy['regions'][str(g)]['rigid_translation']=True
        a,b=old.update(p,w,stamp,legacy);c,d=new.update(p,w,stamp,m)
        if k>=10:
            for dest,out in zip(errors,[a,c]):dest.append(float(np.linalg.norm((np.asarray(out)[[13,15],:2]-p[[13,15],:2])*[640,360],axis=1).max()))
        assert np.array_equal(np.asarray(a)[[12,14,16]],np.asarray(c)[[12,14,16]])
        assert np.array_equal(np.asarray(c)[:,3],p[:,3])
        assert np.isfinite(c).all() and np.isfinite(d).all()
    check(f'{fps} fps supported XY follows current packet',max(errors[1])<1e-10)
    metrics.append({'fps':fps,'old_final_filter_max_error_px':max(errors[0]),'new_final_filter_max_error_px':max(errors[1])})

old=OLD.ArmStabilizer();new=ArmStabilizer();p,w=pose()
old.update(p,w,0,evidence(0));new.update(p,w,0,evidence(0))
p[[13,15],1]+=.03;m=evidence(.1,active=['left'],supported={'left':[2]})
a,_=old.update(p,w,.1,m);c,_=new.update(p,w,.1,m)
check('unsupported wrist keeps original filter',np.array_equal(np.asarray(a)[15],np.asarray(c)[15]))
check('supported elbow bypasses added XY filter',np.allclose(np.asarray(c)[13,:2],p[13,:2],atol=1e-15))

for problem in ['missing_wrist','missing_shoulder','out_of_frame','gap','nan','backward_time']:
    f=ArmStabilizer();p,w=pose();f.update(p,w,1,evidence(1));p[15,1]+=.04;stamp=1.1
    if problem=='missing_wrist':p[15,3]=.1
    if problem=='missing_shoulder':p[11,3]=.1
    if problem=='out_of_frame':p[15,0]=1.2
    if problem=='gap':stamp=2
    if problem=='nan':p[15,0]=np.nan
    if problem=='backward_time':stamp=.9
    a,b=f.update(p,w,stamp,evidence(stamp,active=['left']))
    check(problem+' does not invent observations',np.array_equal(a,p,equal_nan=True))

# Real image-follower + all camera-side locks, including reversal and settling.
class Pipeline:
    def __init__(self):
        self.image=ArmImageFollower();self.repair=ArmTranslationRepair();self.body=LandmarkLock(True);self.quiet=QuietRegionLock();self.sleeve=SleeveLock();self.region=RegionalLock();self.final=ArmStabilizer()
    def run(self,p,w,t,m):
        a,b=self.image.update(p,w,t,m);m=self.image.motion_evidence(m)
        a,b=self.repair.update(a,b,t,m)
        a,b=self.body.update(a,b,t);a,b=self.quiet.update(a,b);a,b=self.sleeve.update(a,b)
        a,b=self.image.retain_motion(a,b,self.body,self.sleeve)
        a,b=self.region.update(a,b,p,t,m);self.image.commit(a,b,t)
        result=self.final.update(a,b,t,m)
        return result,m

pipeline=Pipeline();p0,w=pose();prior_angle=0;active_count=0;max_error=0;settled=[]
for k in range(100):
    t=k/7;angle=0 if k<10 else .5*np.sin((k-10)/20) if k<73 else .5*np.sin(62/20)
    p=p0.copy();s=p0[11,:2]*[640,360]
    def rotation(a):return np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    p[[13,15],:2]=((p0[[13,15],:2]*[640,360]-s)@rotation(angle).T+s)/[640,360]
    step=rotation(angle-prior_angle);affine=np.c_[step,s-step@s];m=evidence(t,quiet=k<10 or k>=73)
    for g in (2,3):
        m['regions'][str(g)].update(reference=k,image_transform=dict(matrix=affine.tolist(),stamp=(k-1)/7,span_ratio=1.,residual_px=0.))
    (a,b),m=pipeline.run(p,w,t,m)
    if 'left' in m.get('arm_follow_active',[]):
        active_count+=1;max_error=max(max_error,float(np.linalg.norm((np.asarray(a)[[13,15],:2]-p[[13,15],:2])*[640,360],axis=1).max()))
    if k>=90:settled.append(np.asarray(a))
    prior_angle=angle
check('integrated motion and reversal use optical support',active_count>40 and max_error<1e-8)
check('return to stillness has no recoil',max(np.max(abs(x-settled[0])) for x in settled)<1e-10)

# Replay recorded DECISIONS, not camera pose: the supplied video has virtual
# clothes baked in, and the logs intentionally do not retain body coordinates.
rows=json.loads((BACKUP/'female_stability_live_before.json').read_text())['recent']
conflicts={}
for side,groups in [('left',(2,3)),('right',(4,5))]:
    before=after=active=0
    for row in rows:
        state=row['arm_image'].get(side,{})
        if not state.get('motion_active'):continue
        active+=1;t=row['captured_at'];m=dict(valid=True,stamp=t,regions=copy.deepcopy(row['image_regions']),arm_follow_active=[side])
        before+=bool(OLD.image_still(m,t,side))
        after+=bool(image_still(m,t,side))
    conflicts[side]={'active_samples':active,'old_false_still_votes':before,'new_false_still_votes':after}
check('recorded contradictory votes removed',sum(x['old_false_still_votes'] for x in conflicts.values())==26 and all(x['new_false_still_votes']==0 for x in conflicts.values()))

report={'checks_passed':len(checks),'checks':checks,'synthetic_filter_response':metrics,
        'recorded_evidence_replay':conflicts,'integrated_active_samples':active_count,'integrated_max_xy_error_px':max_error,
        'limitations':'Offline filter/decision checks, not a raw-video pose replay or a camera-to-display latency measurement. The screen recording contains an existing virtual overlay.'}
Path(__file__).with_name('validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
