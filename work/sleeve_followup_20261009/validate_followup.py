"""Regression reproductions for the recorded motion -> quiet handover."""
import copy,importlib.util,json,sys,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'FahadAR'))
from female_arm_image import ArmImageFollower
from female_arm_stabilizer import ArmStabilizer
from female_arm_observations import ArmTranslationRepair
from female_reliability import RegionalLock
from female_stability import SleeveLock
from stability import LandmarkLock,QuietRegionLock
BACKUP=ROOT/'Backups/sleeve_followup_20261009'
def previous(name):
    spec=importlib.util.spec_from_file_location('previous_'+name,BACKUP/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
OLD_IMAGE=previous('female_arm_image');OLD_FILTER=previous('female_arm_stabilizer')
checks=[]
def check(name,ok):
    assert ok,name
    checks.append(name)
def pose():
    p=np.tile([.5,.5,0.,.99],(33,1))
    for j,xy in {11:(.58,.3),12:(.42,.3),13:(.69,.35),14:(.31,.35),15:(.8,.4),16:(.2,.4),23:(.55,.58),24:(.45,.58),25:(.55,.74),26:(.45,.74),27:(.55,.92),28:(.45,.92)}.items():p[j,:2]=xy
    w=p.copy();w[:,:3]=(p[:,:3]-[.5,.58,0])*[2,2,1];return p,w
def motion(t,quiet=True):
    return dict(valid=True,stamp=t,epoch=1,mode='quiet' if quiet else 'moving',quiet=quiet,
        regions={str(g):dict(valid=True,quiet=quiet,rigid_translation=quiet,shape_preserved=quiet,
            position_epoch=1,position_px=[0.,0.]) for g in range(10)},translation_px=[0.,0.])
def rotate(a):return np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
class Pipeline:
    def __init__(self,old=False):
        self.image=(OLD_IMAGE.ArmImageFollower if old else ArmImageFollower)()
        self.filter=(OLD_FILTER.ArmStabilizer if old else ArmStabilizer)()
        self.body=LandmarkLock(True);self.quiet=QuietRegionLock();self.sleeve=SleeveLock();self.repair=ArmTranslationRepair();self.regions=RegionalLock()
    def run(self,p,w,t,m):
        a,b=self.image.update(p,w,t,m);m=self.image.motion_evidence(m);a,b=self.repair.update(a,b,t,m)
        a,b=self.body.update(a,b,t);a,b=self.quiet.update(a,b);a,b=self.sleeve.update(a,b)
        a,b=self.image.retain_motion(a,b,self.body,self.sleeve);a,b=self.regions.update(a,b,p,t,m)
        self.image.commit(a,b,t);return self.filter.update(a,b,t,m)

# A persistent raw-detector offset after fusion must not reappear when stopping.
handover_results=[]
for side,ids,gs in [('left',[11,13,15],[2,3]),('right',[12,14,16],[4,5])]:
    old=Pipeline(True);new=Pipeline();p0,w=pose();last_angle=0;events=[];old_out=[];new_out=[]
    for k in range(60):
        t=k/7;angle=0 if k<5 else min(k-5,10)*.035
        s=p0[ids[0],:2]*[640,360];true=p0.copy();true[ids[1:],:2]=((p0[ids[1:],:2]*[640,360]-s)@rotate(angle).T+s)/[640,360]
        p=true.copy()
        # Bias starts once the optical follower has established a motion path;
        # increase it at stop to reproduce the logged detector/fusion conflict.
        if k>=8:p[ids[1:],1]+=(2 if k<18 else 7)/360
        m=motion(t,quiet=k<5 or k>=16)
        step=rotate(angle-last_angle);matrix=np.c_[step,s-step@s]
        for g in gs:m['regions'][str(g)].update(reference=k,image_transform=dict(matrix=matrix.tolist(),stamp=(k-1)/7,span_ratio=.8,residual_px=.05))
        a,b=old.run(p,w,t,copy.deepcopy(m));c,d=new.run(p,w,t,copy.deepcopy(m))
        old_out.append(np.asarray(a));new_out.append(np.asarray(c))
        if new.image.state[side]['reason']=='quiet_motion_handover':events.append(k)
        assert np.isfinite(c).all() and np.isfinite(d).all()
        assert np.array_equal(np.asarray(c)[:,3],p[:,3])
        other=[12,14,16] if side=='left' else [11,13,15]
        assert np.array_equal(np.asarray(a)[other],np.asarray(c)[other])
        last_angle=angle
    check(side+' quiet handover exercised',bool(events))
    first=events[0]
    def displacement(seq,start,end):return float(np.linalg.norm((seq[end][ids[1:],:2]-seq[start][ids[1:],:2])*[640,360],axis=1).max())
    old_drift=displacement(old_out,first-1,59);new_drift=displacement(new_out,first-1,59)
    print('handover diagnostic',side,first,old_drift,new_drift,flush=True)
    check(side+' stop does not reintroduce raw-detector offset',new_drift<.01 and old_drift>1)
    check(side+' settled output remains fixed',max(displacement(new_out,45,k) for k in range(46,60))<1e-9)
    handover_results.append(dict(side=side,old_stop_drift_px=old_drift,new_stop_drift_px=new_drift,first_handover_frame=first))

# Quiet and low-speed unknown behavior must remain identical, including noise.
rng=np.random.default_rng(184138)
for mode in ['quiet','unknown_slow','stale_slow','body_translation']:
    a=OLD_FILTER.ArmStabilizer();b=ArmStabilizer()
    for k in range(180):
        p,w=pose();p[[13,15],:2]+=rng.normal(0,.001,(2,2));w[[13,15],:3]+=rng.normal(0,.009,(2,3));t=k/7
        if mode=='body_translation':p[:,:2]+=[.02*np.sin(k/30),.01*np.cos(k/30)]
        m=motion(t)
        if mode=='unknown_slow':m=None
        if mode=='stale_slow':m=motion(t-1)
        ap,aw=a.update(p,w,t,m);bp,bw=b.update(p,w,t,m)
        assert np.array_equal(ap,bp) and np.array_equal(aw,bw),(mode,k)
    check(mode+' unchanged',True)

# Fast visible motion with unavailable optical patches: no prediction, only
# a time-correct filter response to the current accepted landmark observation.
fast=[]
for fps in [5,6.3,7,15,30,60]:
    a=OLD_FILTER.ArmStabilizer();b=ArmStabilizer();error_a=[];error_b=[]
    for k in range(100):
        t=k/fps;p,w=pose();delta=25*math.sin(max(0,t-.5)*6)
        p[13,1]+=delta/720;p[15,1]+=delta/360
        ap,aw=a.update(p,w,t,None);bp,bw=b.update(p,w,t,None)
        error_a.append(float(np.linalg.norm((np.asarray(ap)[[13,15],:2]-p[[13,15],:2])*[640,360],axis=1).max()))
        error_b.append(float(np.linalg.norm((np.asarray(bp)[[13,15],:2]-p[[13,15],:2])*[640,360],axis=1).max()))
        assert np.array_equal(np.asarray(bp)[:,3],p[:,3])
    check(str(fps)+' fps fallback reduces motion error',np.mean(error_b)<np.mean(error_a))
    fast.append(dict(fps=fps,old_mean_xy_error_px=float(np.mean(error_a)),new_mean_xy_error_px=float(np.mean(error_b)),old_max_px=max(error_a),new_max_px=max(error_b)))

# A quiet start has no handover history and retains the old path exactly.
for case in ['quiet_start','loss','gap','depth_change','unsupported_wrist']:
    f=ArmImageFollower();old=OLD_IMAGE.ArmImageFollower();p,w=pose()
    for instance in [f,old]:instance.update(p,w,1,motion(1))
    f.followed={'left':[2,3]};t=1.1;m=motion(t)
    if case=='quiet_start':f.followed={}
    if case=='gap':t=2;m=motion(t)
    if case=='loss':p[15,3]=.1;p[13,3]=.1
    if case=='depth_change':w[15,2]+=.2
    for g in [2,3]:m['regions'][str(g)].update(reference=2,image_transform=dict(matrix=[[1,0,0],[0,1,0]],stamp=1,span_ratio=.8,residual_px=.05))
    if case=='unsupported_wrist':m['regions']['3']['valid']=False
    p[15,1]+=.005
    a,b=f.update(p,w,t,m);c,d=old.update(p,w,t,m)
    check(case+' retains observation safeguards',np.array_equal(np.asarray(a)[15],np.asarray(c)[15]))

# Existing upstream protection still rejects a lone impossible wrist jump.
body=LandmarkLock(True);p,w=pose();body.update(p,w,1);p[15,0]-=.25
a,b=body.update(p,w,1.1)
check('single-frame landmark outlier still rejected',abs(np.asarray(a)[15,0]-.8)<1e-12)

report=dict(checks_passed=len(checks),checks=checks,handover=handover_results,fast_fallback=fast,
    note='Synthetic filter/lock regression checks. No raw wearer-camera replay or measured post-fix display latency.')
Path(__file__).with_name('validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
