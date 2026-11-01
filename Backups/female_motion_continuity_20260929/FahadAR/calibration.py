"""Dimensionless visual calibration, never a tape-measure/body-scan claim.

No RGB images or personal profiles are persisted. A loose/open shirt's outline
is deliberately NOT used to infer the hidden body's chest/waist circumference.
"""
import math
import numpy as np

IDS=[11,12,23,24]

def observed(p,threshold=.45):
    p=np.asarray(p,dtype=float)
    if p.shape!=(33,4) or not np.isfinite(p).all():return None
    return (p[:,3]>=threshold)&(p[:,0]>.005)&(p[:,0]<.995)&(p[:,1]>.005)&(p[:,1]<.995)

class Calibration:
    def __init__(self,full_outfit=False,full_turn=False):
        self.stage='FRONT';self.samples=[];self.ratio=None;self.side_counts=[0,0]
        self.message='1/3 Face front, arms slightly out; show shoulders to hips.'
        self.last_sample=-1e9;self.ratios={};self.first_time=None
        self.full_outfit=full_outfit;self.full_turn=full_turn;self.back_count=0
        self.lower_samples=[];self.lower_ready=not full_outfit
    def update(self,p,src,heading,t,world=None):
        valid=observed(p,.65)
        if valid is None:return
        p=np.asarray(p);src=np.asarray(src)
        if self.stage=='FRONT' and t-self.last_sample>.75:
            # A return after leaving frame needs a fresh stable window, not
            # old samples from the previous stance/person.
            self.samples=[];self.lower_samples=[]
        wrapped=(heading+180)%360-180
        if self.full_outfit and self.stage=='FRONT' and abs(wrapped)<18 and all(valid[[23,24,25,26]]):
            torso_img=np.linalg.norm((p[11,:2]+p[12,:2]-p[23,:2]-p[24,:2])*.5)
            legs=[np.linalg.norm(p[a,:2]-p[b,:2]) for a,b in [(23,25),(24,26)]]
            if torso_img>.05 and min(legs)>.05 and t-self.last_sample>=.07:
                self.lower_samples.append([t,*[v/torso_img for v in legs]])
                self.lower_samples=self.lower_samples[-40:]
        if self.stage=='FRONT':
            if not all(valid[IDS]) or abs(wrapped)>18:
                self.message='1/3 Face front; both shoulders and hips must be clear.';return
            dst=np.c_[(p[IDS,0]-.5)*(32/9),(.5-p[IDS,1])*2]
            sh=abs(src[0,0]-src[1,0]);dh=abs(dst[0,0]-dst[1,0])
            torso=(dst[:2,1].mean()-dst[2:,1].mean())
            height=src[:2,1].mean()-src[2:,1].mean()
            if min(sh,height,torso)<.06 or dh<.13 or abs(dst[0,1]-dst[1,1])>torso*.2:
                self.message='1/3 Stand upright, nearer if needed; camera level.';return
            ratio=(dh/sh)/(torso/height)
            if not .55<ratio<1.65:
                self.message='1/3 Proportions uncertain: face front and adjust camera.';return
            if t-self.last_sample<.07:return
            self.last_sample=t
            self.samples.append((t,ratio,torso))
            self.samples=self.samples[-40:]
            self.message=f'1/3 Hold still: {len(self.samples)}/40 stable samples'
            if len(self.samples)<40 or t-self.samples[0][0]<2.5:return
            values=np.array(self.samples)[:,1:]
            cv=np.std(values,axis=0)/np.mean(values,axis=0)
            if np.max(cv)>.065:
                self.message='1/3 Too much movement; stand still for calibration.';return
            if self.full_outfit:
                if len(self.lower_samples)<24 or t-self.lower_samples[-1][0]>.3:
                    self.message='1/3 Show BOTH hips and knees for shorts calibration.';return
                lower=np.array(self.lower_samples)[:,1:]
                if np.max(np.std(lower,axis=0)/np.mean(lower,axis=0))>.08:
                    self.message='1/3 Hold legs still; shorts proportions are unstable.';return
                self.lower_ready=True
                self.ratios['thigh_screen_to_torso']=np.median(lower,axis=0).tolist()
            self.ratio=float(np.median(values[:,0]));self.stage='SIDES'
            # Optional skeletal ratios only. No circumference guesses.
            w=np.asarray(world,dtype=float) if world is not None else np.empty((0,))
            if w.shape==(33,4) and np.isfinite(w).all():
                length=np.linalg.norm((w[11,:3]+w[12,:3]-w[23,:3]-w[24,:3])*.5)
                for name,a,b in [('shoulders',11,12),('hips',23,24),('upper_arm_left',11,13),('upper_arm_right',12,14),('thigh_left',23,25),('thigh_right',24,26)]:
                    if valid[a] and valid[b]:self.ratios[name]=float(np.linalg.norm(w[a,:3]-w[b,:3])/max(length,.01))
        if self.stage=='SIDES':
            near=observed(p)
            pairs=[(11,23),(12,24)]
            if abs(wrapped)>=60 and abs(wrapped)<=115 and any(near[a] and near[b] for a,b in pairs):
                if t-self.last_sample>=.07:
                    self.last_sample=t;self.side_counts[0 if wrapped<0 else 1]+=1
            if self.full_turn and abs(wrapped)>150 and any(near[a] and near[b] for a,b in pairs) and t-self.last_sample>=.07:
                self.last_sample=t;self.back_count+=1
            a,b=self.side_counts
            self.message=f'2/3 Turn each side slowly: side A {min(a,16)}/16, B {min(b,16)}/16'
            if self.full_turn:self.message+=f', back {min(self.back_count,16)}/16'
            if min(self.side_counts)>=16 and (not self.full_turn or self.back_count>=16):
                self.stage='REVIEW';self.message='3/3 Face front; adjust chest/waist/depth, then Confirm visual fit.'
    def confirm(self):
        if self.stage!='REVIEW':return False
        self.stage='LIVE';self.message='Visual profile locked (not exact body measurements).';return True

def locked_fit(src,p,ratio,previous=None,width=1.,length=1.,x=0.,z=0.):
    """Height-driven projection: no shoulder-width division at profile angles.

    Uses the reliable near-side shoulder/hip pair if the far side is occluded.
    Returns scale/translation and reliable-landmark error; no stale-pose hold.
    """
    valid=observed(p)
    if valid is None:return None
    p=np.asarray(p);src=np.asarray(src)
    dst=np.c_[(p[IDS,0]-.5)*(32/9),(.5-p[IDS,1])*2]
    pairs=[(0,2),(1,3)]
    usable=[(a,b) for a,b in pairs if valid[IDS[a]] and valid[IDS[b]]]
    if not usable:return None
    heights=[]
    for a,b in usable:
        h=src[a,1]-src[b,1];d=dst[a,1]-dst[b,1]
        if h>.07 and d>.055:heights.append(d/h)
    if not heights:return None
    scale=float(np.median(heights))
    if not .07<scale<3:return None
    # Locked proportion prevents 90-degree collapse. A manual depth trim is
    # applied to mesh geometry in body-local coordinates, not screen width.
    sx=scale*ratio*width;sz=scale*length
    indices=sorted(set(i for pair in usable for i in pair))
    weights=np.array([p[IDS[i],3]**2 for i in indices])
    # Pin to the observed shoulder line; averaging hips into translation can
    # pull the collar below shoulders when human/avatar proportions differ.
    shoulders=[a for a,b in usable]
    offset=np.average(dst[shoulders]-src[shoulders]*[sx,sz],axis=0,weights=[p[IDS[i],3]**2 for i in shoulders])
    target=np.r_[sx,sz,offset+[x,z]]
    if previous is not None:
        target=previous*.15+target*.85
    err=float(np.sqrt(np.mean(((src[indices]*target[:2]+target[2:]-dst[indices])*180)**2)))
    return target,err
