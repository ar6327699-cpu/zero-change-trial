"""Independent arms; preserve the established right-arm numerical behavior."""
import copy,importlib.util,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'FahadAR'))
from female_arm_stabilizer import ArmStabilizer
from female_arm_image import ArmImageFollower
from female_arm_observations import ArmTranslationRepair
from female_stability import SleeveLock
from female_reliability import RegionalLock
from stability import LandmarkLock,QuietRegionLock
BACKUP=ROOT/'Backups/left_arm_match_20261009'
def old(name):
    spec=importlib.util.spec_from_file_location('old_'+name,BACKUP/(name+'.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
OLD_FILTER=old('female_arm_stabilizer');OLD_IMAGE=old('female_arm_image')
checks=[]
def check(name,ok):
    assert ok,name
    checks.append(name)
def pose():
    p=np.tile([.5,.5,0.,.99],(33,1))
    for j,xy in {11:(.58,.3),12:(.42,.3),13:(.69,.35),14:(.31,.35),15:(.8,.4),16:(.2,.4),23:(.55,.58),24:(.45,.58),25:(.55,.74),26:(.45,.74),27:(.55,.92),28:(.45,.92)}.items():p[j,:2]=xy
    w=p.copy();w[:,:3]=(p[:,:3]-[.5,.58,0])*[2,2,1];return p,w
def motion(t,left=(),right=(),quiet=False):
    m=dict(valid=True,stamp=t,epoch=1,mode='quiet' if quiet else 'moving',quiet=quiet,
           regions={str(g):dict(valid=True,quiet=quiet,rigid_translation=quiet,
                    position_epoch=1,position_px=[0.,0.]) for g in range(10)},translation_px=[0.,0.])
    m['arm_follow_active']=(["left"] if left else [])+(["right"] if right else [])
    m['arm_follow_joints']={'left':list(left),'right':list(right)}
    return m

response=[]
for fps in [5,6.3,7,15,30,60]:
    a=OLD_FILTER.ArmStabilizer();b=ArmStabilizer();errors=[[],[]];match=[]
    for k in range(150):
        t=k/fps;p,w=pose();step=np.array([3*math.sin(t*2),10*math.sin(t*3)]) if k>=8 else np.zeros(2)
        p[[13,15],:2]+=step/[640,360];p[[14,16],:2]+=step*[-1,1]/[640,360]
        m=motion(t,left=[13] if k>=8 else [],right=[14,16] if k>=8 else [],quiet=k<8)
        ap,aw=a.update(p,w,t,m);bp,bw=b.update(p,w,t,m);ap=np.asarray(ap);bp=np.asarray(bp)
        if k>=8:
            errors[0].append(float(np.linalg.norm((ap[15,:2]-p[15,:2])*[640,360])))
            errors[1].append(float(np.linalg.norm((bp[15,:2]-p[15,:2])*[640,360])))
            match.append(float(np.linalg.norm((bp[15,:2]-[1-bp[16,0],bp[16,1]])*[640,360])))
        assert np.array_equal(ap[[12,14,16]],bp[[12,14,16]])
        assert np.array_equal(np.asarray(aw),np.asarray(bw))
        assert np.array_equal(bp[:,3],p[:,3])
    check(str(fps)+' fps left common travel matches supported right',max(errors[1])<1e-9 and max(match)<1e-9)
    check(str(fps)+' fps removes reproduced partial-support wrist lag',max(errors[0])>.5)
    response.append(dict(fps=fps,old_left_max_error_px=max(errors[0]),new_left_max_error_px=max(errors[1]),left_vs_mirrored_right_max_px=max(match)))

# Full support, quiet, stale, missing observation and opposite-arm cases.
rng=np.random.default_rng(190927)
for case in ['quiet','no_image','stale','full_support','only_right_moves','missing_elbow','missing_wrist','outside_frame']:
    a=OLD_FILTER.ArmStabilizer();b=ArmStabilizer()
    for k in range(80):
        p,w=pose();t=k/7
        p[[13,15],:2]+=rng.normal(0,.001,(2,2));w[[13,15],:3]+=rng.normal(0,.009,(2,3))
        p[[14,16],1]+=math.sin(k/10)*.04
        m=motion(t,left=[13],right=[14,16],quiet=False)
        if case=='quiet':m=motion(t,quiet=True)
        if case=='no_image':m=None
        if case=='stale':m['stamp']=t-1
        if case=='full_support':m['arm_follow_joints']['left']=[13,15]
        if case=='only_right_moves':m['arm_follow_joints']['left']=[];m['arm_follow_active']=['right']
        if case=='missing_elbow':p[13,3]=.1
        if case=='missing_wrist':p[15,3]=.1
        if case=='outside_frame':p[15,0]=1.2
        ap,aw=a.update(p,w,t,m);bp,bw=b.update(p,w,t,m)
        assert np.array_equal(ap,bp) and np.array_equal(aw,bw),(case,k)
    check(case+' keeps original output',True)

# Random right-arm conditions, including partial observations and loss: the
# requested good side and all its filter diagnostics must remain unchanged.
a=OLD_FILTER.ArmStabilizer();b=ArmStabilizer()
for k in range(250):
    p,w=pose();t=k*.15
    p[[13,14,15,16],:2]+=rng.normal(0,.01,(4,2));w[[13,14,15,16],:3]+=rng.normal(0,.03,(4,3))
    m=motion(t,left=[13],right=[14,16] if k%3==0 else [14] if k%3==1 else [])
    if k%9==0:p[16,3]=.2
    ap,aw=a.update(p,w,t,m);bp,bw=b.update(p,w,t,m)
    assert np.array_equal(np.asarray(ap)[[12,14,16]],np.asarray(bp)[[12,14,16]])
    assert np.array_equal(np.asarray(aw)[[12,14,16]],np.asarray(bw)[[12,14,16]])
    assert a.state['right']==b.state['right']
check('250 mixed frames preserve right output and diagnostics exactly',True)

# A visible wrist may still bend independently of its carried elbow.
f=ArmStabilizer();p,w=pose();f.update(p,w,0,motion(0,quiet=True));start=p.copy()
p[[13,15],1]+=.02;p[15,0]-=.02
a,b=f.update(p,w,.15,motion(.15,left=[13]));a=np.asarray(a)
carried_only=start[15,:2]+[0,.02]
check('left wrist articulation is filtered rather than replaced by elbow',a[15,0]<carried_only[0] and a[15,0]>p[15,0])
check('right cannot drive left pose',np.array_equal(a[[12,14,16]],p[[12,14,16]]))

def image_motion(t,angle,prior_angle,quiet=False):
    p,w=pose();m=motion(t,quiet=quiet)
    for ids,gs,sign in [([11,13,15],[2,3],1),([12,14,16],[4,5],-1)]:
        s=p[ids[0],:2]*[640,360]
        def rot(a):return np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
        r=rot(angle*sign);p[ids[1:],:2]=((p[ids[1:],:2]*[640,360]-s)@r.T+s)/[640,360]
        r=rot((angle-prior_angle)*sign);matrix=np.c_[r,s-r@s]
        for g in gs:m['regions'][str(g)].update(reference=round(t*10),image_transform=dict(matrix=matrix.tolist(),stamp=round(t-.1,6),span_ratio=.7,residual_px=.05))
    return p,w,m

for case in ['corroborated','xy_disagrees','partial_support','missing_joint','stale','gap']:
    a=OLD_IMAGE.ArmImageFollower();b=ArmImageFollower()
    p,w,m=image_motion(1,0,0,True);a.update(p,w,1,m);b.update(p,w,1,m)
    p,w,m=image_motion(1.1,.035,0);a.update(p,w,1.1,m);b.update(p,w,1.1,m)
    p,w,m=image_motion(1.2,.07,.035);w[15,2]+=.12;w[16,2]+=.12;t=1.2
    if case=='xy_disagrees':p[15,0]+=.01
    if case=='partial_support':m['regions']['3']['valid']=False
    if case=='missing_joint':p[15,3]=.1
    if case=='stale':m['stamp']=0
    if case=='gap':t=2;m['stamp']=t
    ap,aw=a.update(p,w,t,m);bp,bw=b.update(p,w,t,m)
    assert np.array_equal(np.asarray(ap)[[12,14,16]],np.asarray(bp)[[12,14,16]])
    assert a.state.get('right')==b.state.get('right')
    assert np.array_equal(aw,bw) and np.array_equal(bw,w)
    if case=='corroborated':
        check('bounded left depth jitter retains verified image tracking',b.state['left'].get('depth_guard')=='current_xy_corroborated' and 'left' in b.moving and a.state['left']['reason']=='depth_changed')
    else:check(case+' does not bypass depth/visibility safeguards',b.state.get('left',{}).get('depth_guard') is None)

class Pipeline:
    def __init__(self,old=False):
        self.image=(OLD_IMAGE.ArmImageFollower if old else ArmImageFollower)()
        self.filter=(OLD_FILTER.ArmStabilizer if old else ArmStabilizer)()
        self.body=LandmarkLock(True);self.quiet=QuietRegionLock();self.sleeve=SleeveLock()
        self.repair=ArmTranslationRepair();self.regions=RegionalLock()
    def run(self,p,w,t,m):
        a,b=self.image.update(p,w,t,m);m=self.image.motion_evidence(m)
        a,b=self.repair.update(a,b,t,m);a,b=self.body.update(a,b,t)
        a,b=self.quiet.update(a,b);a,b=self.sleeve.update(a,b)
        a,b=self.image.retain_motion(a,b,self.body,self.sleeve)
        a,b=self.regions.update(a,b,p,t,m);self.image.commit(a,b,t)
        return self.filter.update(a,b,t,m)

# Integrated camera lock chain: common forearm translation around the elbow,
# partial left image evidence, independent right evidence, then quiet settling.
a=Pipeline(True);b=Pipeline();p0,w=pose();last_angle=0;old_errors=[];new_errors=[];settled=[]
for k in range(120):
    t=k/7;angle=0 if k<8 else .25*math.sin((k-8)/6) if k<88 else .25*math.sin(79/6)
    p=p0.copy();m=motion(t,quiet=k<8 or k>=88)
    for ids,gs,sign in [([11,13,15],[2,3],1),([12,14,16],[4,5],-1)]:
        s=p0[ids[0],:2]*[640,360];elbow=p0[ids[1],:2]*[640,360]
        def rot(v):return np.array([[math.cos(v),-math.sin(v)],[math.sin(v),math.cos(v)]])
        moved=(elbow-s)@rot(angle*sign).T+s;shift=moved-elbow
        p[ids[1:],:2]+=shift/[640,360]
        r=rot((angle-last_angle)*sign);upper=np.c_[r,s-r@s]
        for g in gs:m['regions'][str(g)].update(reference=k,image_transform=dict(matrix=upper.tolist(),stamp=(k-1)/7,span_ratio=.8,residual_px=.05))
        # Forearm translates with its elbow without imposing the upper-arm
        # rotation on wrist articulation. Its optical step is independent.
        prior=(elbow-s)@rot(last_angle*sign).T+s;d=moved-prior
        m['regions'][str(gs[1])]['image_transform']['matrix']=[[1.,0.,float(d[0])],[0.,1.,float(d[1])]]
    if 8<=k<88:m['regions']['3']['valid']=False
    ap,aw=a.run(p,w,t,copy.deepcopy(m));bp,bw=b.run(p,w,t,copy.deepcopy(m));ap=np.asarray(ap);bp=np.asarray(bp)
    assert np.array_equal(ap[[12,14,16]],bp[[12,14,16]])
    assert np.isfinite(bp).all() and np.isfinite(bw).all()
    if b.filter.state['left'].get('wrist_carried_by_elbow'):
        old_errors.append(float(np.linalg.norm((ap[15,:2]-p[15,:2])*[640,360])))
        new_errors.append(float(np.linalg.norm((bp[15,:2]-p[15,:2])*[640,360])))
    if k>=105:settled.append(bp)
    last_angle=angle
check('integrated partial-support chain improves left wrist response',len(new_errors)>40 and np.mean(new_errors)<np.mean(old_errors)*.35)
check('integrated right arm remains unchanged',True)
check('integrated left settles without continued creep',max(np.max(abs(p-settled[0])) for p in settled)<1e-9)
integrated=dict(samples=len(new_errors),old_mean_left_error_px=float(np.mean(old_errors)),new_mean_left_error_px=float(np.mean(new_errors)))

report=dict(checks_passed=len(checks),checks=checks,common_forearm_travel=response,integrated_camera_chain=integrated,
    right_arm_numerically_unchanged=True,independent_arm_motion=True,
    limitations='Synthetic filter/image-evidence tests. No post-change live wearer video or raw-camera replay; missing forearm articulation remains conservatively filtered.')
Path(__file__).with_name('validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
