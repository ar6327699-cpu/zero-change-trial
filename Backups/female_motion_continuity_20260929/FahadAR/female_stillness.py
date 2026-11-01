"""Female-only final-frame stillness gate. No pose prediction or RGB freezing.

The fixed observation anchor bounds drift. Genuine accumulated motion, depth
motion, confidence changes, settings changes and missing packets release it.
This complements joint filters: a quiet input must not let downstream solver
settling or projection fitting continue to move the rendered garment.
"""
import numpy as np

IDS=np.array([11,12,13,14,15,16,23,24,25,26,27,28])

class StillnessGate:
    def __init__(self):self.reset()
    def reset(self):
        self.anchor=None;self.started=None;self.last=None;self.context=None

    def update(self,pose,world,stamp,context=None):
        p=np.asarray(pose,float);w=np.asarray(world,float);stamp=float(stamp)
        if p.shape!=(33,4) or w.shape!=(33,4) or not np.isfinite(p).all() or not np.isfinite(w).all() or not np.isfinite(stamp):
            self.reset();return False
        valid=(p[IDS,3]>=.55)&(p[IDS,0]>.005)&(p[IDS,0]<.995)&(p[IDS,1]>.005)&(p[IDS,1]<.995)
        # At least one observed shoulder/hip pair and enough limb evidence.
        if valid.sum()<8 or not ((valid[0] and valid[6]) or (valid[1] and valid[7])):
            self.reset();return False
        if self.last is not None and not 0<stamp-self.last<=.35:self.reset()
        if context!=self.context:self.reset()
        self.context=context;self.last=stamp
        if self.anchor is not None:
            pp,ww,vv=self.anchor
            if np.array_equal(valid,vv):
                idx=IDS[valid]
                pixels=np.linalg.norm((p[idx,:2]-pp[idx,:2])*[640,360],axis=1)
                # World landmarks are pelvis-relative. Compare relative
                # vectors to reject inferred root-depth translation noise.
                centre=w[[23,24],:3].mean(0);old_centre=ww[[23,24],:3].mean(0)
                depth=np.linalg.norm((w[idx,:3]-centre)-(ww[idx,:3]-old_centre),axis=1)
                if pixels.max()<=3.0 and depth.max()<=.06:
                    return stamp-self.started>=.28
        self.anchor=(p.copy(),w.copy(),valid.copy());self.started=stamp
        return False
