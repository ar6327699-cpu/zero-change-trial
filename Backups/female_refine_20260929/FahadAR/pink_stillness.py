"""Pink-only fixed-anchor stillness gate with confidence hysteresis.

Never freezes the camera. The render path releases on observed joint travel,
body turn/depth travel, stale packets, lost near-side evidence or fit changes.
Other outfits keep their existing gate.
"""
import numpy as np

PINK_TOP='e915c671a1b65e0cb0c7b4a4e08cc629'
IDS=np.array([11,12,13,14,15,16,23,24,25,26,27,28])

class StillnessGate:
    def __init__(self):self.reset()
    def reset(self):
        self.anchor=None;self.started=None;self.last=None;self.context=None
        self.seen=np.zeros(12,dtype=bool);self.missing_since=None

    def update(self,pose,world,stamp,context=None):
        p=np.asarray(pose,float);w=np.asarray(world,float);stamp=float(stamp)
        if p.shape!=(33,4) or w.shape!=(33,4) or not np.isfinite(p).all() or not np.isfinite(w).all() or not np.isfinite(stamp):
            self.reset();return False
        if (self.last is not None and not 0<stamp-self.last<=.5) or context!=self.context:self.reset()
        self.context=context;self.last=stamp
        # Confidence near .55 must not repeatedly switch a stationary limb on
        # and off. Acquiring uses .60, retaining uses .45; actual motion of an
        # acquired joint is still evaluated against the FIXED anchor.
        threshold=np.where(self.seen,.45,.60)
        valid=(p[IDS,3]>=threshold)&(p[IDS,0]>.005)&(p[IDS,0]<.995)&(p[IDS,1]>.005)&(p[IDS,1]<.995)
        self.seen=valid.copy()
        near_left=valid[[0,2,4,6,8,10]].sum()
        near_right=valid[[1,3,5,7,9,11]].sum()
        evidence=valid.sum()>=6 and ((valid[0] and valid[6] and near_left>=4) or (valid[1] and valid[7] and near_right>=4))
        if not evidence:
            self.reset();return False
        if self.anchor is not None:
            pp,ww,vv=self.anchor
            common=valid&vv
            lost=vv&~valid
            if lost.any():
                if self.missing_since is None:self.missing_since=stamp
            else:self.missing_since=None
            # A briefly hidden far joint cannot destabilize the near side;
            # prolonged loss reacquires, never pins an unseen limb forever.
            grace=self.missing_since is None or stamp-self.missing_since<=.2
            if common.sum()>=6 and grace and not (valid&~vv).any():
                idx=IDS[common]
                pixels=np.linalg.norm((p[idx,:2]-pp[idx,:2])*[640,360],axis=1)
                centre=w[[23,24],:3].mean(0);old=ww[[23,24],:3].mean(0)
                depth=np.linalg.norm((w[idx,:3]-centre)-(ww[idx,:3]-old),axis=1)
                if pixels.max()<=3.0 and depth.max()<=.06:
                    return stamp-self.started>=.28
        self.anchor=(p.copy(),w.copy(),valid.copy());self.started=stamp;self.missing_since=None
        return False
