"""Final female garment hold with image-corroborated bounded hysteresis.

The conservative landmark-only gate remains the fallback for old senders or
textureless/occluded patches. Valid image evidence may veto inferred-depth
noise, but cannot override observed large travel, stale time or lost torso.
"""
import numpy as np
from pink_stillness import StillnessGate as LandmarkGate, IDS

class StillnessGate:
    def __init__(self): self.reset()
    def reset(self):
        self.fallback=LandmarkGate();self.anchor=None;self.context=None
        self.last=None;self.started=None;self.epoch=None
        self.missing_since=None

    def update(self,pose,world,stamp,context=None,motion=None):
        p=np.asarray(pose,float);w=np.asarray(world,float);stamp=float(stamp)
        if p.shape!=(33,4) or w.shape!=(33,4) or not np.isfinite(p).all() or not np.isfinite(w).all() or not np.isfinite(stamp):
            self.reset();return False
        if context!=self.context or (self.last is not None and not 0<stamp-self.last<=.6):self.reset()
        self.context=context;self.last=stamp
        fallback=self.fallback.update(p,w,stamp,context)
        visual=isinstance(motion,dict) and motion.get('valid') is True and abs(float(motion.get('stamp',-1))-stamp)<.001
        if not visual:
            self.anchor=None;self.started=None;self.epoch=None
            return fallback
        # A reliable moving-image observation is an explicit veto, even if
        # an upstream low-pass filter has not yet moved its output landmarks.
        if not motion.get('quiet') and motion.get('mode')!='translate':
            self.anchor=None;self.started=None;self.fallback.reset();return False
        shift=np.asarray(motion.get('translation_px',[0,0]),float)
        if shift.shape!=(2,) or not np.isfinite(shift).all() or np.linalg.norm(shift)>12:
            self.reset();return False
        threshold=.45 if self.anchor is not None else .60
        valid=(p[IDS,3]>=threshold)&np.all((p[IDS,:2]>.005)&(p[IDS,:2]<.995),axis=1)
        evidence=valid.sum()>=6 and ((valid[0] and valid[6]) or (valid[1] and valid[7]))
        if not evidence:self.reset();return False
        epoch=motion.get('epoch')
        if self.anchor is not None and self.epoch==epoch:
            pp,ww,vv,prior_shift=self.anchor;common=valid&vv
            if (vv&~valid).any():
                if self.missing_since is None:self.missing_since=stamp
            else:self.missing_since=None
            grace=self.missing_since is None or stamp-self.missing_since<=.2
            if common.sum()>=6 and grace and not (valid&~vv).any():
                idx=IDS[common]
                pixel=np.linalg.norm((p[idx,:2]-pp[idx,:2])*[640,360]-(shift-prior_shift),axis=1)
                depth=np.linalg.norm((w[idx,:3]-w[[23,24],:3].mean(0))-(ww[idx,:3]-ww[[23,24],:3].mean(0)),axis=1)
                # Fixed bounds, not frame-to-frame bounds: slow motion cannot
                # walk an anchor indefinitely. Optical travel is limited to
                # 1.25px in quiet mode; coherent small whole-body translation
                # is carried by the root without re-solving garment shape.
                covered=set(motion.get('covered',[]))
                limits=np.array([6.0 if int(j) in covered else 3.0 for j in idx])
                # A single monocular wrist-depth estimate can swing while
                # its RGB patch stays fixed. Bound that outlier separately
                # from coherent whole-body 3D travel/turns.
                if np.all(pixel<=limits) and depth.max()<=.18 and np.median(depth)<=.06:
                    return stamp-self.started>=.28
        self.anchor=(p.copy(),w.copy(),valid.copy(),shift.copy());self.started=stamp;self.epoch=epoch
        self.missing_since=None
        return False
