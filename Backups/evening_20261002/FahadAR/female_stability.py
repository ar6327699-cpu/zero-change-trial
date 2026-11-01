"""Female-only confidence and residual continuity; no camera-frame prediction."""
import math
import numpy as np

PARENT={11:12,12:11,23:24,24:23,13:11,14:12,15:13,16:14,25:23,26:24,27:25,28:26}

class ResidualLock:
    def __init__(self):self.reset()
    def reset(self):self.values={};self.seen={};self.time=None
    def update(self,measured,stamp):
        stamp=float(stamp);dt=.05 if self.time is None else max(.005,min(.25,stamp-self.time));self.time=stamp
        previous=self.values;result={}
        for i,value in measured.items():
            value=np.asarray(value,float);old=previous.get(i)
            if old is not None:
                distance=np.linalg.norm(value-old)*180
                # Fixed-anchor deadband rejects subpixel shimmer but releases
                # cumulative slow movement. Fast observed motion remains prompt.
                alpha=0. if distance<.65 else 1-math.exp(-dt/(.035 if distance<3 else .006))
                value=old+(value-old)*alpha
            result[i]=value.copy();self.seen[i]=stamp
        # Resolve torso, then elbow/knee, then wrist/ankle. Never jump straight
        # to zero as a joint crosses the detector's visibility threshold.
        for i in (11,12,23,24,13,14,25,26,15,16,27,28):
            if i in result:continue
            parent=PARENT[i];base=result.get(parent,np.zeros(3));old=previous.get(i)
            age=stamp-self.seen.get(i,-1e9)
            if old is not None and age<.45:
                relative=old-previous.get(parent,np.zeros(3))
                # Decay only after a short dropout; no indefinite frozen limbs.
                decay=1. if age<=.18 else math.exp(-dt/.09)
                result[i]=base+relative*decay
            else:result[i]=base.copy()
        self.values={i:v.copy() for i,v in result.items()}
        return result

class SleeveLock:
    """Quiet sleeve depth follows a fixed anchor, independently of torso motion."""
    def __init__(self):self.anchors={}
    def reset(self):self.anchors={}
    def update(self,pose,world):
        p=np.asarray(pose,float).copy();w=np.asarray(world,float).copy()
        if p.shape!=(33,4) or w.shape!=(33,4):return pose,world
        for ids in ((11,13,15),(12,14,16)):
            idx=list(ids)
            if np.min(p[idx,3])<.55:self.anchors.pop(ids,None);continue
            old=self.anchors.get(ids)
            if old is not None:
                pp,ww=old
                # Translation-invariant: a quiet arm can travel with the body.
                rel=p[idx,:2]-p[idx[0],:2];prior=pp[:,:2]-pp[0,:2]
                depth=(w[idx,:3]-w[idx[0],:3])-(ww[:,:3]-ww[0,:3])
                if np.max(np.linalg.norm((rel-prior)*[640,360],axis=1))<1.25 and np.max(np.linalg.norm(depth,axis=1))<.035:
                    p[idx[1:],:3]=pp[1:,:3]+p[idx[0],:3]-pp[0,:3]
                    w[idx[1:],:3]=ww[1:,:3]+w[idx[0],:3]-ww[0,:3]
                    continue
            self.anchors[ids]=(p[idx].copy(),w[idx].copy())
        return p.tolist(),w.tolist()
