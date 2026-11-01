"""Deterministic, confidence-aware body lock shared by Eastern profiles.

No mesh rebuilding and no cloth simulation. Time is supplied by the capture
packet so tests and live playback use the same filtering rules.
"""
import math
import numpy as np

EASTERN={'KURTA','SHALWAR','PAJAMA','FEMALE_KAMEEZ','FEMALE_TROUSER','FEMALE_SHALWAR','FEMALE_DUPATTA'}

class QuietRegionLock:
    """Hold a quiet torso/leg against a fixed image-space anchor, not last frame.

    This also holds inferred depth: invisible depth noise must not wobble a
    stationary neckline. A real slow move accumulates past the anchor and
    releases the region; confidence loss never invents visible observations.
    """
    def __init__(self):self.reset()
    def reset(self):self.anchors={}
    def update(self,pose,world):
        p=np.asarray(pose,float).copy();w=np.asarray(world,float).copy()
        for ids in ((11,12,23,24),(25,27),(26,28)):
            key=ids;idx=list(ids)
            if np.any(p[idx,3]<.55):self.anchors.pop(key,None);continue
            previous=self.anchors.get(key)
            if previous is not None:
                pp,ww=previous
                distance=np.linalg.norm((p[idx,:2]-pp[:,:2])*[640,360],axis=1)
                if distance.max()<=1.75:
                    p[idx,:3]=pp[:,:3];w[idx,:3]=ww[:,:3]
                    continue
            self.anchors[key]=(p[idx].copy(),w[idx].copy())
        return p.tolist(),w.tolist()

class LandmarkLock:
    def __init__(self):self.reset()
    def reset(self):
        self.previous=None;self.stamp=None;self.seen=np.full(33,-1e9)
        self.world=None;self.candidate=None;self.joint_candidates={}
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
        world_translation=np.median(w[ids,:3]-self.world[ids,:3],axis=0)
        for i in range(33):
            if valid[i]:
                residual=p[i,:2]-old[i,:2]-translation
                pixels=np.linalg.norm(residual*np.array([640,360]))
                candidate=self.joint_candidates.get(i)
                outlier=pixels>90 and stamp-self.seen[i]<.3
                confirmed=candidate is not None and np.linalg.norm((p[i,:2]-candidate)*[640,360])<35
                if outlier and not confirmed:
                    self.joint_candidates[i]=p[i,:2].copy()
                    out[i,:2]=old[i,:2]+translation
                    wo[i,:3]=self.world[i,:3]+world_translation
                else:
                    self.joint_candidates.pop(i,None)
                    delta=p[i,:3]-old[i,:3]
                    # A stationary torso does not imply stationary hands/feet.
                    # Filter each joint's motion; fast limbs must meet this RGB.
                    distance=np.linalg.norm(delta[:2]*[640,360])
                    tau=.075 if distance<1.2 else .022 if distance<4 else .006
                    alpha=1-math.exp(-dt/tau)
                    if distance<.35:delta[:2]=0
                    out[i,:3]=old[i,:3]+alpha*delta
                    wo[i,:3]=self.world[i,:3]+alpha*(w[i,:3]-self.world[i,:3])
                    self.seen[i]=stamp
                out[i,3]=p[i,3];wo[i,3]=w[i,3]
            elif stamp-self.seen[i]<.6:
                out[i,:2]+=translation;out[i,3]=max(.56,old[i,3])
                wo[i,:3]+=world_translation
            else:out[i,3]=p[i,3]
        self.previous=out;self.world=wo;self.stamp=stamp
        return out.tolist(),wo.tolist()

class FitLock:
    def __init__(self):self.value=None;self.time=None;self.scale_samples=[]
    def update(self,target,stamp,anchor=None):
        target=np.asarray(target,float)
        if target.shape!=(4,) or not np.isfinite(target).all() or min(target[:2])<=0:return self.value
        if self.value is None:self.value=target.copy();self.time=stamp;return target.copy()
        dt=float(np.clip(stamp-self.time,.005,.25));self.time=stamp;old=self.value
        anchor=np.zeros(2) if anchor is None else np.asarray(anchor,float)
        old_position=old[:2]*anchor+old[2:]
        position=target[:2]*anchor+target[2:]
        self.scale_samples.append(target[:2].copy());self.scale_samples=self.scale_samples[-3:]
        relative=target[:2]/old[:2]-1
        moving=np.max(abs(relative))>.025
        scale=target[:2] if moving else np.median(self.scale_samples,axis=0)
        # Adapt promptly to distance changes; the former 12%/second cap made
        # a step towards the camera leave the garment behind for seconds.
        alpha=1-math.exp(-dt/(.025 if moving else .14))
        if np.max(abs(relative))>.003:old[:2]+=alpha*(scale-old[:2])
        delta=position-old_position;pixels=np.linalg.norm(delta*180)
        alpha=1-math.exp(-dt/(.055 if pixels<1.2 else .008))
        position=old_position+alpha*delta if pixels>.25 else old_position
        # Translation and scale refer to the SAME shoulder anchor. Filtering
        # a new-scale offset independently caused sliding during zoom/motion.
        old[2:]=position-old[:2]*anchor
        return old.copy()

def packet_valid(meta,pixels,previous_seq=-1):
    if meta.get('kind')!='frame':return False
    if not isinstance(meta.get('seq'),int) or meta['seq']<=previous_seq:return False
    if len(pixels)!=640*360*4 or meta.get('pixel_format')!='rgb+matte8':return False
    if not math.isfinite(float(meta.get('captured_at',float('nan')))):return False
    return True
