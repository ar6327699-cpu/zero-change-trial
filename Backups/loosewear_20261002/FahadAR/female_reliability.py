"""Female-only stillness and regional evidence; shared male gates stay intact.

All anchors are fixed, not creeping averages. Confidence is never increased.
Image evidence can corroborate uncertain joints briefly, not forever.
"""
from collections import Counter
import numpy as np
from pink_stillness import IDS, StillnessGate as LandmarkGate

LOSS_SECONDS = .65
REGIONS = {'torso': (11,12,23,24), 'left_arm': (11,13,15),
           'right_arm': (12,14,16), 'left_leg': (23,25,27),
           'right_leg': (24,26,28)}
GROUPS = {'torso': (0,1), 'left_arm': (2,3), 'right_arm': (4,5),
          'left_leg': (6,7), 'right_leg': (8,9)}

def fresh(motion, stamp):
    try:
        return (isinstance(motion,dict) and motion.get('valid') is True and
                abs(float(motion.get('stamp',-1))-float(stamp))<.001)
    except (TypeError,ValueError):
        return False

class StillnessGate:
    def __init__(self):
        self.counts=Counter();self.reason='new';self.reset()
    def reset(self):
        self.fallback=LandmarkGate();self.anchor=None;self.context=None
        self.last=None;self.started=None;self.epoch=None
        self.acquired=np.zeros(len(IDS),bool);self.missing=np.full(len(IDS),np.nan)
    def decide(self, held, reason):
        self.reason=reason;self.counts[reason]+=1;return bool(held)
    def diagnostics(self):
        return {'reason':self.reason,'counts':dict(self.counts),'missing_grace_ms':LOSS_SECONDS*1000}
    def update(self,pose,world,stamp,context=None,motion=None):
        p=np.asarray(pose,float);w=np.asarray(world,float);stamp=float(stamp)
        if p.shape!=(33,4) or w.shape!=(33,4) or not np.isfinite(p).all() or not np.isfinite(w).all() or not np.isfinite(stamp):
            self.reset();return self.decide(False,'invalid_pose')
        if context!=self.context or (self.last is not None and not 0<stamp-self.last<=.6):self.reset()
        self.context=context;self.last=stamp
        fallback=self.fallback.update(p,w,stamp,context)
        if not fresh(motion,stamp):
            self.anchor=None;self.started=None;self.epoch=None
            return self.decide(fallback,'landmark_hold' if fallback else 'image_unknown')
        if not motion.get('quiet') and motion.get('mode')!='translate':
            self.anchor=None;self.started=None;self.fallback.reset()
            return self.decide(False,'observed_movement')
        shift=np.asarray(motion.get('translation_px',[0,0]),float)
        if shift.shape!=(2,) or not np.isfinite(shift).all() or np.linalg.norm(shift)>12:
            self.reset();return self.decide(False,'translation_limit')
        threshold=.45 if self.anchor is not None else .60
        valid=(p[IDS,3]>=threshold) & np.all((p[IDS,:2]>.005)&(p[IDS,:2]<.995),axis=1)
        lost=self.acquired&~valid
        self.missing[lost & np.isnan(self.missing)]=stamp
        self.missing[valid]=np.nan
        self.acquired |= valid
        evidence=valid.sum()>=6 and ((valid[0] and valid[6]) or (valid[1] and valid[7]))
        if not evidence:
            self.reset();return self.decide(False,'torso_or_joints_missing')
        # Keep the missing timer across re-anchors. Repeated unknown packets
        # must not renew a stale limb's lease or freeze an unseen body forever.
        if np.any(lost & ((stamp-self.missing)>LOSS_SECONDS)):
            self.anchor=None;self.started=None
            return self.decide(False,'joint_loss_expired')
        epoch=motion.get('epoch')
        reason='new_anchor'
        if self.anchor is not None and self.epoch==epoch:
            pp,ww,vv,prior_shift=self.anchor
            # Keep the successful original acquisition behavior. Short loss
            # never removed a joint from this anchor, so its return is common;
            # a genuinely new visible joint still requires reacquisition.
            common=valid&vv
            returning=valid&~vv
            if returning.any():
                reason='reacquired_joint_moved'
            if common.sum()>=6 and reason!='reacquired_joint_moved':
                idx=IDS[common]
                pixel=np.linalg.norm((p[idx,:2]-pp[idx,:2])*[640,360]-(shift-prior_shift),axis=1)
                depth=np.linalg.norm((w[idx,:3]-w[[23,24],:3].mean(0))-(ww[idx,:3]-ww[[23,24],:3].mean(0)),axis=1)
                covered=set(motion.get('covered',[]))
                limits=np.array([6. if int(j) in covered else 3. for j in idx])
                if np.all(pixel<=limits) and depth.max()<=.18 and np.median(depth)<=.06:
                    held=stamp-self.started>=.28
                    return self.decide(held,'visual_hold' if held else 'settling')
                reason='joint_travel' if np.any(pixel>limits) else 'depth_travel'
        elif self.anchor is not None:reason='image_anchor_changed'
        self.anchor=(p.copy(),w.copy(),valid.copy(),shift.copy())
        self.started=stamp;self.epoch=epoch
        return self.decide(False,reason)

class RegionalLock:
    """Stabilize independent fixed regions before pose solving.

Moving regions immediately use current data. A quiet sleeve is body-local,
so holding it never prevents the shoulder itself from following the person.
"""
    def __init__(self):self.reset()
    def reset(self):self.anchors={};self.missing={};self.last=None;self.held=[]
    def update(self,pose,world,observed,stamp,motion):
        p=np.asarray(pose,float).copy();w=np.asarray(world,float).copy();raw=np.asarray(observed,float)
        self.held=[]
        if p.shape!=(33,4) or w.shape!=(33,4) or raw.shape!=(33,4) or not np.isfinite(p).all() or not np.isfinite(w).all() or not np.isfinite(raw).all():
            self.reset();return pose,world
        if self.last is not None and not 0<stamp-self.last<=.6:self.reset()
        self.last=stamp
        regions=motion.get('regions',{}) if isinstance(motion,dict) and abs(float(motion.get('stamp',-1))-stamp)<.001 else {}
        for name,indices in REGIONS.items():
            idx=list(indices);old=self.anchors.get(name)
            evidence=[regions.get(str(g),{}) for g in GROUPS[name]]
            # Never promote missing imagery to quiet. Use the existing input
            # unchanged when a region has insufficient visual corroboration.
            quiet=bool(evidence) and all(e.get('valid') and e.get('quiet') for e in evidence)
            visible=raw[idx,3]>=.45
            if not visible.all():self.missing.setdefault(name,stamp)
            else:self.missing.pop(name,None)
            expired=stamp-self.missing.get(name,stamp)>LOSS_SECONDS
            if not quiet or expired or not visible[0]:
                self.anchors.pop(name,None);continue
            optical=[(e.get('position_epoch'),np.asarray(e.get('position_px',[0,0]),float)) for e in evidence]
            if old is not None:
                pp,ww,rp,prior_optical=old
                optical_quiet=all(a[0]==b[0] and np.linalg.norm(a[1]-b[1])<=1.25 for a,b in zip(optical,prior_optical))
                if name=='torso':
                    delta=(raw[idx,:2]-rp[:,:2])*[640,360]
                    depth=(w[idx,:3]-w[idx,:3].mean(0))-(ww[:,:3]-ww[:,:3].mean(0))
                else:
                    delta=((raw[idx,:2]-raw[idx[0],:2])-(rp[:,:2]-rp[0,:2]))*[640,360]
                    depth=(w[idx,:3]-w[idx[0],:3])-(ww[:,:3]-ww[0,:3])
                if optical_quiet and np.linalg.norm(delta[visible],axis=1).max()<=6 and np.linalg.norm(depth[visible],axis=1).max()<=.18:
                    if name=='torso':p[idx,:3]=pp[:,:3];w[idx,:3]=ww[:,:3]
                    else:
                        p[idx[1:],:3]=pp[1:,:3]+p[idx[0],:3]-pp[0,:3]
                        w[idx[1:],:3]=ww[1:,:3]+w[idx[0],:3]-ww[0,:3]
                    self.held.append(name);continue
            self.anchors[name]=(p[idx].copy(),w[idx].copy(),raw[idx].copy(),optical)
        return p.tolist(),w.tolist()
