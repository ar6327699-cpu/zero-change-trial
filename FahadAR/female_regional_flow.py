"""Independent fixed image references for female torso and limb segments.

Moving one sleeve must not repeatedly re-sample a still sleeve or integrate
patch-selection noise into its position. Unknown imagery is never held quiet.
"""
import cv2
import numpy as np
from female_motion import SEGMENTS
from female_arm_features import PatchSelector
from female_reliability import LOSS_SECONDS
from female_flow_stats import percentile

class ImageMotion:
    def __init__(self):self.reset()
    def reset(self):
        self.selector=PatchSelector();self.refs={};self.base={};self.positions={}
        self.position_epoch={};self.missing={};self.last=None;self.epoch=0
        self.reference_stamps={}
        self.arm_settle={}
        self.origin={};self.translation_base=np.zeros(2);self.serial=0
        self.required=();self.points=np.empty((0,1,2),np.float32)
    def _anchor(self,gray,p,stamp,groups=None):
        wanted=set(range(10) if groups is None else groups)
        self.selector.required=tuple(self.refs)
        # Independent references only consume the groups being renewed.
        # Selecting discarded groups again wastes CPU but supplies no evidence.
        self.selector._anchor(gray,p,stamp,groups=wanted)
        self.serial+=1;reference=gray.copy()
        for g in wanted:
            selected=self.selector.labels==g
            if np.count_nonzero(selected)>=4:
                self.refs[g]=(self.serial,reference,self.selector.points[selected].copy())
                self.reference_stamps[g]=stamp
                self.base[g]=self.positions.get(g,np.zeros(2)).copy()
            else:
                if g in self.refs:self.position_epoch[g]=self.position_epoch.get(g,0)+1
                self.refs.pop(g,None);self.base.pop(g,None);self.positions.pop(g,None)
        self.required=tuple(sorted(self.refs))
        self.points=np.concatenate([r[2] for r in self.refs.values()]) if self.refs else np.empty((0,1,2),np.float32)
        self.last=stamp
    def update(self,rgb,pose,stamp):
        p=np.asarray(pose,float);stamp=float(stamp)
        if p.shape!=(33,4) or not np.isfinite(p).all() or not np.isfinite(stamp):
            self.reset();return dict(valid=False,quiet=False,stamp=stamp,epoch=self.epoch,regions={},reason='invalid_pose')
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        if gray.shape!=(360,640):
            self.reset();return dict(valid=False,quiet=False,stamp=stamp,epoch=self.epoch,regions={},reason='invalid_image')
        if self.last is None or not 0<stamp-self.last<=.6:
            self.reset();self.epoch=1;self._anchor(gray,p,stamp)
            return dict(valid=False,quiet=False,stamp=stamp,epoch=self.epoch,regions={},reason='new_anchor')
        self.last=stamp;matches={};groups={};buckets={}
        for g,(serial,reference,pts) in self.refs.items():
            buckets.setdefault(serial,[]).append((g,reference,pts))
        for rows in buckets.values():
            pts=np.concatenate([r[2] for r in rows]);labels=np.concatenate([np.full(len(r[2]),r[0]) for r in rows])
            reference=rows[0][1]
            # Full-resolution crops with a pyramid/window guard. Coordinates
            # stay in camera pixels; neither tracking nor video is downscaled.
            ends=p[[j for g,_,_ in rows for j in SEGMENTS[g]],:2]*[640,360]
            ext=np.concatenate([pts[:,0],ends]);lo=np.maximum(np.floor(ext.min(0)-64),0).astype(int)
            hi=np.minimum(np.ceil(ext.max(0)+65),[640,360]).astype(int);x0,y0=lo;x1,y1=hi
            src=pts-lo.astype(np.float32);a=reference[y0:y1,x0:x1];b=gray[y0:y1,x0:x1]
            forward,status,_=cv2.calcOpticalFlowPyrLK(a,b,src,None,winSize=(21,21),maxLevel=2)
            if forward is None:continue
            backward,backstatus,_=cv2.calcOpticalFlowPyrLK(b,a,forward,None,winSize=(21,21),maxLevel=2)
            if backward is None:continue
            good=(status[:,0]>0)&(backstatus[:,0]>0)&(np.linalg.norm(backward[:,0]-src[:,0],axis=1)<.7)
            forward+=lo.astype(np.float32)
            for g,_,original in rows:
                if min(p[list(SEGMENTS[g]),3])<.45:self.missing.setdefault(g,stamp)
                else:self.missing.pop(g,None)
                selected=labels==g;valid=selected&good
                ok=stamp-self.missing.get(g,stamp)<=LOSS_SECONDS and valid.sum()>=4 and valid.sum()>=selected.sum()*.65
                groups[str(g)]=dict(valid=bool(ok),quiet=False)
                if ok:matches[g]=(pts[valid,0],forward[valid,0])
        absolute={};all_delta={};settled_renew=set()
        for g,(src,dst) in matches.items():
            delta=dst-src;position=self.base[g]+np.median(delta,axis=0)
            self.positions[g]=position
            absolute[g]=position-self.origin.get(g,np.zeros(2))
            all_delta[g]=delta+self.base[g]-self.origin.get(g,np.zeros(2))
        torso=[absolute[g] for g in matches if g<2]
        translation=np.median(torso,axis=0) if torso else np.zeros(2)
        residuals=[];travel=[]
        for g,(src,dst) in matches.items():
            delta=dst-src
            distance=float(percentile(np.linalg.norm(delta,axis=1),80))
            residual=float(percentile(np.linalg.norm(all_delta[g]-translation,axis=1),80))
            affine,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=.5,maxIters=100,confidence=.99,refineIters=5)
            rigid=False;shape_preserved=False
            if affine is not None and inliers is not None and inliers.mean()>=.8:
                ends=p[list(SEGMENTS[g]),:2]*[640,360];axis=ends[1]-ends[0];length=np.linalg.norm(axis)
                span=np.ptp(src@(axis/max(length,1.)))
                endpoint_delta=ends@affine[:,:2].T+affine[:,2]-ends+self.base[g]-self.origin.get(g,np.zeros(2))
                rigid=span>=length*.25 and np.max(np.linalg.norm(endpoint_delta-translation,axis=1))<.6
                # Segment shape can stay unchanged while another limb makes
                # the whole-person translation estimate non-rigid. Expose a
                # LOCAL test too; it is not a vote that the image is still.
                local_translation=self.positions[g]-self.origin.get(g,np.zeros(2))
                shape_preserved=span>=length*.25 and np.max(np.linalg.norm(endpoint_delta-local_translation,axis=1))<.6
            groups[str(g)].update(quiet=distance<=1.25,flow_px=distance,residual_px=residual,
                position_px=self.positions[g].tolist(),position_epoch=self.position_epoch.get(g,0),
                rigid_translation=bool(rigid),shape_preserved=bool(shape_preserved),reference=self.refs[g][0])
            if g in (2,3,4,5) and affine is not None and inliers is not None and inliers.mean()>=.8:
                # Measured current-frame displacement, independent of the
                # detector's next elbow/wrist estimate. Existing hold votes
                # above are intentionally unchanged.
                reprojection=np.linalg.norm(src@affine[:,:2].T+affine[:,2]-dst,axis=1)
                groups[str(g)]['image_transform']={
                    'matrix':affine.tolist(),'stamp':self.reference_stamps[g],
                    'span_ratio':float(span/max(length,1.)),
                    'residual_px':float(percentile(reprojection[inliers[:,0]>0],90))}
                # Once an articulated sleeve STOPS, a displaced old reference
                # can leave its midpoint non-quiet or its rigid vote false
                # indefinitely. Verify a fixed image transform for .32s, then
                # refresh only this segment. Never declare unknown pixels
                # quiet, never freeze a pose, and do not integrate tiny steps
                # into a perpetually renewed stillness test.
                fit=groups[str(g)]['image_transform']
                eligible=not rigid and fit['span_ratio']>=.4 and fit['residual_px']<=.5
                if eligible:
                    old=self.arm_settle.get(g);reference=self.refs[g][0]
                    if old is not None and old[0]==reference:
                        fixed=old[2];observed=fixed@affine[:,:2].T+affine[:,2]
                        stable=np.linalg.norm(observed-old[3],axis=1).max()<=.2
                    else:stable=False
                    if not stable:
                        self.arm_settle[g]=(reference,stamp,ends.copy(),ends@affine[:,:2].T+affine[:,2])
                    elif stamp-old[1]>=.32:
                        settled_renew.add(g);self.arm_settle.pop(g,None)
                else:self.arm_settle.pop(g,None)
            elif g in (2,3,4,5):self.arm_settle.pop(g,None)
            residuals.append(residual);travel.append(distance)
        enough=bool(torso) and sum(g>=2 for g in matches)>=2
        coherent=enough and max(residuals)<=1.25
        # Returning to the original body position can make cumulative travel
        # zero while a renewed image reference is still 10px away. That is
        # coherent translation, not articulation or a new pose anchor.
        mode='translate' if coherent and (np.linalg.norm(translation)>.65 or max(travel)>1.25) else 'quiet' if enough and max(travel)<=1.25 else 'moving'
        result=dict(valid=enough,quiet=enough and mode=='quiet',mode=mode,stamp=stamp,epoch=self.epoch,
            regions=groups,translation_px=(self.translation_base+translation).tolist(),
            max_flow_px=max(travel,default=0),region_count=len(matches),
            covered=sorted({j for g in matches for j in SEGMENTS[g]}),reason=mode if enough else 'insufficient_regions')
        for g in (2,3,4,5):
            if g not in matches:self.arm_settle.pop(g,None)
        renew={g for g in range(10) if g not in matches}|settled_renew
        if mode=='moving':renew.update(g for g in matches if not groups[str(g)]['quiet'])
        elif coherent:renew.update(g for g in matches if groups[str(g)]['flow_px']>10)
        # Only a region that needs renewed evidence loses its image reference.
        # Other regions keep the same original pixels and features exactly.
        if renew:self._anchor(gray,p,stamp,renew)
        if mode=='moving' or not enough:
            self.origin={g:v.copy() for g,v in self.positions.items()}
            self.translation_base=np.zeros(2);self.epoch+=1
        for g in settled_renew:
            if g in self.positions:
                shift=translation if enough and mode!='moving' else np.zeros(2)
                self.origin[g]=self.positions[g]-shift
        return result
