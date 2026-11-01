"""Continuous female sleeve stabilizer: the final arm stage before the solver.

Upstream sleeve locks are binary: a held sleeve is frozen, a released one jumps
to the current detector estimate. At ~7 processed fps that hold/release
alternation, plus monocular depth noise, is the visible sleeve tremble. This
stage filters elbow/wrist RELATIVE to the already-filtered shoulder, so body
translation and the torso pipeline pass through untouched and the sleeve
stays on the wearer.

Adaptive (One Euro style) cutoff driven by the filtered VELOCITY vector:
alternating detector noise cancels, coherent real motion persists, so a real
swing still opens the filter promptly. Elbow and wrist share one cutoff per
arm, so differential lag cannot bend the forearm.

The two arms are not equally observable: one sleeve can carry twice the
detector noise of the other. Each arm therefore learns its own noise speed
while fixed image patches confirm that sleeve is still (quiet, or rigidly
translating with the body), and its motion threshold scales with that noise.
Image-confirmed stillness also suppresses detector travel directly. Image
evidence preserves the existing quiet/unknown path. Confirmed optical
articulation follows the current fused screen observation without another
low-pass/deadzone; only supported joints take that path. Inferred depth
remains filtered, avoiding both contradictory still votes and cascaded XY lag.
Confidence is never raised, missing joints are never invented, and nothing
predicts future frames.
"""
import math
import numpy as np
from female_reliability import GROUPS, fresh

ARMS={'left':(11,13,15),'right':(12,14,16)}
PIXELS=np.array([640.,360.])

def alpha(cutoff,dt):
    return 1/(1+1/(2*math.pi*cutoff*dt))

def soft_deadzone(value,anchor,width):
    """Hold inside width; beyond it follow continuously, never snap."""
    delta=value-anchor;distance=float(np.linalg.norm(delta))
    if distance<=width:return anchor.copy()
    return anchor+delta*(1-width/distance)

def image_still(motion,stamp,side):
    """True only when BOTH sleeve patches are valid and quiet/rigid; else None."""
    if not fresh(motion,stamp):return None
    if side in motion.get('arm_follow_active',[]):return None
    evidence=[motion.get('regions',{}).get(str(g),{}) for g in GROUPS[side+'_arm']]
    if not all(e.get('valid') for e in evidence):return None
    return True if all(e.get('quiet') or e.get('rigid_translation') for e in evidence) else None

class ArmStabilizer:
    # Cutoffs in Hz; speeds in px/s of the 640x360 frame, shoulder-relative.
    MIN_CUTOFF=.55;BETA=.08;MAX_CUTOFF=8.;SPEED_CUTOFF=4.
    NOISE_FLOOR=10.;NOISE_GAIN=1.;NOISE_CUTOFF=.4;NOISE_MAX=60.
    STILL_CUTOFF=.3
    DEPTH_MIN_CUTOFF=.3;DEPTH_BETA=.025
    DEADZONE_PX=1.;STILL_DEADZONE_PX=1.5;DEADZONE_M=.015;FADE_SPEED=45.

    def __init__(self):self.reset()
    def reset(self):
        self.joints={};self.arms={};self.last=None;self.state={}

    def update(self,pose,world,stamp,motion=None):
        p=np.asarray(pose,float).copy();w=np.asarray(world,float).copy()
        try:stamp=float(stamp)
        except (TypeError,ValueError):stamp=float('nan')
        if p.shape!=(33,4) or w.shape!=(33,4) or not np.isfinite(p).all() or not np.isfinite(w).all() or not math.isfinite(stamp):
            self.reset();return pose,world
        continuous=self.last is not None and 0<stamp-self.last<=.5
        if not continuous:
            # Keep each arm's learned noise level; only filter state restarts.
            self.joints={};self.arms={k:dict(noise=v['noise']) for k,v in self.arms.items()}
        dt=stamp-self.last if continuous else None
        self.last=stamp;self.state={}
        seen=(p[:,3]>=.5)&np.all((p[:,:2]>.01)&(p[:,:2]<.99),axis=1)
        for side,(s,*ids) in ARMS.items():
            joints=[j for j in ids if seen[j]] if seen[s] else []
            for j in ids:
                if j not in joints:self.joints.pop(j,None)
            arm=self.arms.setdefault(side,dict(noise=self.NOISE_FLOOR))
            if not joints:
                arm.pop('speed',None);self.state[side]={'filtered':False,'reason':'unobserved'};continue
            rel={j:(p[j,:2]-p[s,:2])*PIXELS for j in joints}
            depth={j:w[j,:3]-w[s,:3] for j in joints}
            if dt is None or 'speed' not in arm or any(j not in self.joints for j in joints):
                # (Re)acquisition starts from the current observation exactly.
                for j in joints:self.joints[j]=dict(xy=rel[j].copy(),out=rel[j].copy(),velocity=np.zeros(2),
                                                    world=depth[j].copy(),wout=depth[j].copy())
                arm['speed']=0.;self.state[side]={'filtered':False,'reason':'acquired'};continue
            av=alpha(self.SPEED_CUTOFF,dt)
            for j in joints:
                q=self.joints[j];q['velocity']=q['velocity']+av*((rel[j]-q['xy'])/dt-q['velocity'])
            speed=max(float(np.linalg.norm(self.joints[j]['velocity'])) for j in joints)
            still=image_still(motion,stamp,side)
            following=set(motion.get('arm_follow_joints',{}).get(side,[])) if (
                fresh(motion,stamp) and side in motion.get('arm_follow_active',[])) else set()
            if still:
                # Patches moved <=1.25px: this detector travel is noise.
                arm['noise']=float(np.clip(arm['noise']+alpha(self.NOISE_CUTOFF,dt)*(speed-arm['noise']),self.NOISE_FLOOR,self.NOISE_MAX))
            floor=max(self.NOISE_FLOOR,self.NOISE_GAIN*arm['noise'])
            excess=0. if still else max(0.,speed-floor)
            arm['speed']=speed
            low=self.STILL_CUTOFF if still else self.MIN_CUTOFF
            a=alpha(min(self.MAX_CUTOFF,low+self.BETA*excess),dt)
            ad=alpha(min(self.MAX_CUTOFF,self.DEPTH_MIN_CUTOFF+self.DEPTH_BETA*excess),dt)
            fade=max(0.,1-excess/self.FADE_SPEED)
            dead=(self.STILL_DEADZONE_PX if still else self.DEADZONE_PX)*fade
            for j in joints:
                q=self.joints[j]
                if j in following:
                    # The image follower already fused detector innovation.
                    # A second low-pass lags THIS packet's RGB. Synchronize
                    # both hidden states so settling cannot spring backward.
                    q['xy']=rel[j].copy();q['out']=rel[j].copy()
                else:
                    q['xy']=q['xy']+a*(rel[j]-q['xy'])
                    q['out']=soft_deadzone(q['xy'],q['out'],dead)
                q['world']=q['world']+ad*(depth[j]-q['world'])
                q['wout']=soft_deadzone(q['world'],q['wout'],self.DEADZONE_M*fade)
                p[j,:2]=p[s,:2]+q['out']/PIXELS;w[j,:3]=w[s,:3]+q['wout']
            self.state[side]={'filtered':True,'speed_px_s':speed,'noise_px_s':arm['noise'],
                              'image_still':bool(still),'alpha':a,'depth_alpha':ad,
                              'following_joints':sorted(following.intersection(joints)),
                              'xy_lag_px':float(max(np.linalg.norm(rel[j]-self.joints[j]['out']) for j in joints))}
        return p.tolist(),w.tolist()
