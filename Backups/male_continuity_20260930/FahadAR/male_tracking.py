"""Male reference outfits only: fixed-anchor render hold and observed neck ROI.

No changes to garment assets or Female filters. A hold is bounded by current
landmark evidence; it is not a frozen camera or an unlimited lost-body pose.
"""
import numpy as np

IDS=np.array([11,12,13,14,15,16,23,24,25,26,27,28])

class StillnessGate:
    def __init__(self):self.reset()
    def reset(self):
        self.anchor=None;self.started=None;self.last=None;self.context=None
        self.seen=np.zeros(len(IDS),dtype=bool);self.missing_since=None
    def update(self,pose,world,stamp,context=None):
        p=np.asarray(pose,float);w=np.asarray(world,float);stamp=float(stamp)
        if p.shape!=(33,4) or w.shape!=(33,4) or not np.isfinite(p).all() or not np.isfinite(w).all() or not np.isfinite(stamp):
            self.reset();return False
        if (self.last is not None and not 0<stamp-self.last<=.5) or context!=self.context:self.reset()
        self.context=context;self.last=stamp
        threshold=np.where(self.seen,.45,.60)
        valid=(p[IDS,3]>=threshold)&(p[IDS,0]>.005)&(p[IDS,0]<.995)&(p[IDS,1]>.005)&(p[IDS,1]<.995)
        self.seen=valid.copy()
        # A profile needs observed same-side torso, arm and leg controls.
        # Never use the invisible far side as confidence for a hold.
        near=any(valid[a] and valid[h] and valid[list(side)].sum()>=4
                 for a,h,side in [(0,6,(0,2,4,6,8,10)),(1,7,(1,3,5,7,9,11))])
        if valid.sum()<6 or not near:self.reset();return False
        if self.anchor is not None:
            pp,ww,vv=self.anchor;common=valid&vv;lost=vv&~valid
            if lost.any():
                if self.missing_since is None:self.missing_since=stamp
            else:self.missing_since=None
            grace=self.missing_since is None or stamp-self.missing_since<=.2
            if common.sum()>=6 and grace and not (valid&~vv).any():
                idx=IDS[common]
                pixels=np.linalg.norm((p[idx,:2]-pp[idx,:2])*[640,360],axis=1)
                # Use observed torso to remove world translation, not a hidden hip.
                core=[i for i in (11,12,23,24) if i in idx]
                centre=w[core,:3].mean(0);old=ww[core,:3].mean(0)
                depth=np.linalg.norm((w[idx,:3]-centre)-(ww[idx,:3]-old),axis=1)
                if pixels.max()<=2.25 and depth.max()<=.045:
                    return stamp-self.started>=.25
        self.anchor=(p.copy(),w.copy(),valid.copy());self.started=stamp;self.missing_since=None
        return False

def neck_roi(pose):
    """Conservative CURRENT observed nape/neck region, including face dropout.

    Requires both torso and at least one visible ear. It only proposes a ROI;
    current skin evidence still decides its alpha. Never paints a skin patch.
    """
    import cv2
    p=np.asarray(pose,float);out=np.zeros((360,640),np.uint8)
    if p.shape!=(33,4) or not np.isfinite(p).all():return out
    seen=(p[:,3]>=.5)&(p[:,0]>.005)&(p[:,0]<.995)&(p[:,1]>.005)&(p[:,1]<.995)
    pairs=[(a,b) for a,b in ((11,23),(12,24)) if seen[a] and seen[b]]
    ears=[i for i in (7,8) if seen[i]]
    if not pairs or not ears:return out
    xy=p[:,:2]*[640,360];torso=np.median([np.linalg.norm(xy[a]-xy[b]) for a,b in pairs])
    if torso<18:return out
    # Single near shoulder is lateral to the neck; centre from observed ears.
    ear=xy[ears].mean(0);shoulder_y=float(np.mean([xy[a,1] for a,b in pairs]))
    height=shoulder_y-ear[1]
    if not torso*.08<height<torso*.65:return out
    half=float(np.clip(torso*.085,2,14))
    top=ear+np.array([0,height*.40]);bottom=np.array([ear[0],shoulder_y-torso*.035])
    pts=np.rint([top+[-half*.8,0],top+[half*.8,0],bottom+[half,0],bottom+[-half,0]]).astype(np.int32)
    cv2.fillConvexPoly(out,pts,255)
    return out
