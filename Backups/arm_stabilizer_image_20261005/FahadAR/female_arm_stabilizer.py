"""Continuous female sleeve stabilizer: the final arm stage before the solver.

Upstream sleeve locks are binary: a held sleeve is frozen, a released one jumps
to the current detector estimate. At ~7 processed fps that hold/release
alternation, plus monocular depth noise, is the visible sleeve tremble. This
stage filters elbow/wrist RELATIVE to the already-filtered shoulder, so body
translation and the torso pipeline pass through untouched and the sleeve
stays on the wearer.

Adaptive (One Euro style) cutoff: a still arm gets strong smoothing; measured
2D arm speed above a noise floor raises the cutoff so a real swing follows
promptly. Elbow and wrist share one cutoff per arm, so differential lag cannot
bend the forearm. A small continuous deadzone removes residual shimmer without
the stick-slip jump of a hard hold; it fades out as the arm speeds up. Depth,
the least observable axis, uses a lower cutoff driven by the same 2D speed.
Confidence is never raised, missing joints are never invented, and nothing
predicts future frames.
"""
import math
import numpy as np

ARMS={'left':(11,13,15),'right':(12,14,16)}
PIXELS=np.array([640.,360.])

def alpha(cutoff,dt):
    return 1/(1+1/(2*math.pi*cutoff*dt))

def soft_deadzone(value,anchor,width):
    """Hold inside width; beyond it follow continuously, never snap."""
    delta=value-anchor;distance=float(np.linalg.norm(delta))
    if distance<=width:return anchor.copy()
    return anchor+delta*(1-width/distance)

class ArmStabilizer:
    # Cutoffs in Hz; speed is the filtered max elbow/wrist speed in px/s of
    # the 640x360 frame, relative to the shoulder. Tuned on recorded wearer
    # footage: halves still-arm jitter, ~3px median lag during fast swings.
    MIN_CUTOFF=.55;BETA=.05;MAX_CUTOFF=8.;NOISE_FLOOR=12.;SPEED_CUTOFF=4.
    DEPTH_MIN_CUTOFF=.3;DEPTH_BETA=.025
    DEADZONE_PX=1.;DEADZONE_M=.015;FADE_SPEED=45.

    def __init__(self):self.reset()
    def reset(self):
        self.joints={};self.arms={};self.last=None;self.state={}

    def update(self,pose,world,stamp):
        p=np.asarray(pose,float).copy();w=np.asarray(world,float).copy()
        try:stamp=float(stamp)
        except (TypeError,ValueError):stamp=float('nan')
        if p.shape!=(33,4) or w.shape!=(33,4) or not np.isfinite(p).all() or not np.isfinite(w).all() or not math.isfinite(stamp):
            self.reset();return pose,world
        continuous=self.last is not None and 0<stamp-self.last<=.5
        if not continuous:self.joints={};self.arms={}
        dt=stamp-self.last if continuous else None
        self.last=stamp;self.state={}
        seen=(p[:,3]>=.5)&np.all((p[:,:2]>.01)&(p[:,:2]<.99),axis=1)
        for side,(s,*ids) in ARMS.items():
            joints=[j for j in ids if seen[j]] if seen[s] else []
            for j in ids:
                if j not in joints:self.joints.pop(j,None)
            if not joints:
                self.arms.pop(side,None);self.state[side]={'filtered':False,'reason':'unobserved'};continue
            rel={j:(p[j,:2]-p[s,:2])*PIXELS for j in joints}
            depth={j:w[j,:3]-w[s,:3] for j in joints}
            speed=self.arms.get(side)
            if dt is None or speed is None or any(j not in self.joints for j in joints):
                # (Re)acquisition starts from the current observation exactly.
                for j in joints:self.joints[j]=dict(xy=rel[j].copy(),out=rel[j].copy(),world=depth[j].copy(),wout=depth[j].copy())
                self.arms[side]=0.;self.state[side]={'filtered':False,'reason':'acquired'};continue
            raw_speed=max(float(np.linalg.norm(rel[j]-self.joints[j]['xy']))/dt for j in joints)
            speed+=alpha(self.SPEED_CUTOFF,dt)*(raw_speed-speed);self.arms[side]=speed
            excess=max(0.,speed-self.NOISE_FLOOR)
            a=alpha(min(self.MAX_CUTOFF,self.MIN_CUTOFF+self.BETA*excess),dt)
            ad=alpha(min(self.MAX_CUTOFF,self.DEPTH_MIN_CUTOFF+self.DEPTH_BETA*excess),dt)
            fade=max(0.,1-speed/self.FADE_SPEED)
            for j in joints:
                q=self.joints[j]
                q['xy']=q['xy']+a*(rel[j]-q['xy'])
                q['out']=soft_deadzone(q['xy'],q['out'],self.DEADZONE_PX*fade)
                q['world']=q['world']+ad*(depth[j]-q['world'])
                q['wout']=soft_deadzone(q['world'],q['wout'],self.DEADZONE_M*fade)
                p[j,:2]=p[s,:2]+q['out']/PIXELS;w[j,:3]=w[s,:3]+q['wout']
            self.state[side]={'filtered':True,'speed_px_s':speed,'alpha':a,'depth_alpha':ad}
        return p.tolist(),w.tolist()
