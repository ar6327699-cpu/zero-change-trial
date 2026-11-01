"""Deterministic, confidence-aware body lock shared by Eastern profiles.

No mesh rebuilding and no cloth simulation. Time is supplied by the capture
packet so tests and live playback use the same filtering rules.
"""
import math
import numpy as np

EASTERN={'KURTA','SHALWAR','PAJAMA','FEMALE_KAMEEZ','FEMALE_TROUSER','FEMALE_SHALWAR','FEMALE_DUPATTA'}

class LandmarkLock:
    def __init__(self):self.reset()
    def reset(self):self.previous=None;self.stamp=None;self.seen=np.full(33,-1e9);self.world=None;self.candidate=None
    def update(self,pose,world,stamp):
        p=np.asarray(pose,float);w=np.asarray(world,float)
        if p.shape!=(33,4) or w.shape!=(33,4) or not np.isfinite(p).all() or not np.isfinite(w).all():return None
        dt=.05 if self.stamp is None else float(np.clip(stamp-self.stamp,.005,.25))
        valid=(p[:,3]>=.5)&(p[:,0]>-.03)&(p[:,0]<1.03)&(p[:,1]>-.03)&(p[:,1]<1.03)
        torso=valid[[11,12,23,24]]
        if not (torso[:2].any() and torso[2:].any()):return None
        if self.previous is None:
            self.previous=p.copy();self.world=w.copy();self.seen[valid]=stamp;self.stamp=stamp;return p.tolist(),w.tolist()
        old=self.previous;out=old.copy();wo=self.world.copy()
        common=valid[[11,12,23,24]]
        ids=np.array([11,12,23,24])[common]
        translation=np.median(p[ids,:2]-old[ids,:2],axis=0)
        # A one-frame teleport is held. Two consistent measurements permit a
        # genuine rapid movement, without permanently capping its position.
        if np.linalg.norm(translation*np.array([640,360]))>75:
            centre=p[ids,:2].mean(0)
            if self.candidate is None or np.linalg.norm(centre-self.candidate)>.06:
                self.candidate=centre;return None
        self.candidate=None
        speed=np.linalg.norm(translation*np.array([640,360]))/dt
        alpha=1-math.exp(-dt/(.09 if speed<18 else .035 if speed<100 else .012))
        for i in range(33):
            if valid[i]:
                residual=p[i,:2]-old[i,:2]-translation
                if np.linalg.norm(residual*np.array([640,360]))>90 and stamp-self.seen[i]<.3:
                    out[i,:2]=old[i,:2]+translation
                else:
                    delta=p[i,:3]-old[i,:3]
                    if np.linalg.norm(delta[:2]*[640,360])<.55:delta[:2]=0
                    out[i,:3]=old[i,:3]+alpha*delta
                    wo[i,:3]=self.world[i,:3]+alpha*(w[i,:3]-self.world[i,:3])
                    self.seen[i]=stamp
                out[i,3]=p[i,3];wo[i,3]=w[i,3]
            elif stamp-self.seen[i]<.6:
                out[i,:2]+=translation;out[i,3]=max(.56,old[i,3])
            else:out[i,3]=p[i,3]
        self.previous=out;self.world=wo;self.stamp=stamp
        return out.tolist(),wo.tolist()

class FitLock:
    def __init__(self):self.value=None;self.time=None;self.scale_samples=[]
    def update(self,target,stamp):
        target=np.asarray(target,float)
        if target.shape!=(4,) or not np.isfinite(target).all() or min(target[:2])<=0:return self.value
        if self.value is None:self.value=target.copy();self.time=stamp;return target.copy()
        dt=float(np.clip(stamp-self.time,.005,.25));self.time=stamp;old=self.value
        self.scale_samples.append(target[:2].copy());self.scale_samples=self.scale_samples[-5:]
        scale=np.median(self.scale_samples,axis=0)
        # Do not let pose shortening or a noisy hip breathe the garment size.
        relative=scale/old[:2]-1
        if len(self.scale_samples)>=4 and np.max(abs(relative))>.012:
            old[:2]*=1+np.clip(relative,-.12*dt,.12*dt)
        delta=target[2:]-old[2:];pixels=np.linalg.norm(delta*180)
        if pixels>.45:
            alpha=1-math.exp(-dt/(.07 if pixels<2 else .025))
            old[2:]+=alpha*delta
        return old.copy()

def packet_valid(meta,pixels,previous_seq=-1):
    if meta.get('kind')!='frame':return False
    if not isinstance(meta.get('seq'),int) or meta['seq']<=previous_seq:return False
    if len(pixels)!=640*360*4 or meta.get('pixel_format')!='rgb+matte8':return False
    if not math.isfinite(float(meta.get('captured_at',float('nan')))):return False
    return True
