"""Female-only optical evidence and bounded foreground processing cost."""
import cv2
import numpy as np
from female_motion import ImageMotion as OriginalImageMotion, SEGMENTS
from female_reliability import LOSS_SECONDS
from occlusion import MaskStabilizer as OriginalMaskStabilizer

class ImageMotion(OriginalImageMotion):
    def reset(self):
        super().reset();self.missing={};self.reason='new'
        self.regional_base={};self.regional_epoch={}
    def _anchor(self,gray,p,stamp):
        # Refreshing a moving sleeve must not discard a still torso just
        # because an already tracked joint fluctuated from .60 to .48.
        # The .45 continuation threshold matches update(); NEW regions still
        # need .60. This selects RGB patches; detector confidence is unchanged.
        previous=set(self.required)
        points=[];labels=[];required=[];xy=p[:,:2]*[640,360]
        for group,(a,b) in enumerate(SEGMENTS):
            if min(p[a,3],p[b,3])<(.45 if group in previous else .6):continue
            if not np.all((xy[[a,b]]>[4,4]) & (xy[[a,b]]<[636,356])):continue
            length=np.linalg.norm(xy[a]-xy[b])
            if length<9:continue
            radius=int(np.clip(length*(.10 if group<2 else .07),2,7))
            mask=np.zeros(gray.shape,np.uint8)
            u=xy[a]*.8+xy[b]*.2;v=xy[a]*.2+xy[b]*.8
            cv2.line(mask,tuple(u.astype(int)),tuple(v.astype(int)),255,2*radius)
            low=np.maximum(np.floor(np.minimum(u,v)-radius-2).astype(int),0)
            high=np.minimum(np.ceil(np.maximum(u,v)+radius+3).astype(int),[640,360])
            x0,y0=low;x1,y1=high
            found=cv2.goodFeaturesToTrack(gray[y0:y1,x0:x1],18,.025,3,mask=mask[y0:y1,x0:x1],blockSize=3)
            if found is None or len(found)<4:continue
            found[:,0,:]+=low
            points.extend(found[:,0,:]);labels.extend([group]*len(found));required.append(group)
        self.gray=gray.copy();self.points=np.asarray(points,np.float32).reshape(-1,1,2)
        self.labels=np.asarray(labels);self.required=tuple(required);self.last=stamp;self.epoch+=1
    def update(self,rgb,pose,stamp):
        stamp=float(stamp);p=np.asarray(pose,float)
        no=dict(valid=False,quiet=False,stamp=stamp,epoch=self.epoch,regions={})
        if p.shape!=(33,4) or not np.isfinite(p).all() or not np.isfinite(stamp):
            self.reset();return dict(no,reason='invalid_pose')
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        if gray.shape!=(360,640):self.reset();return dict(no,reason='invalid_image')
        if self.last is None or not 0<stamp-self.last<=.6 or len(self.points)<12:
            if self.last is None or not 0<stamp-self.last<=.6:self.required=()
            self._anchor(gray,p,stamp);self.missing={};self.regional_base={};self.regional_epoch={}
            return dict(no,reason='new_anchor')
        self.last=stamp
        # An uncertain pose joint does not instantly destroy independently
        # matched RGB evidence. Require FB-checked points and a finite grace.
        usable=[]
        for g in self.required:
            if min(p[list(SEGMENTS[g]),3])<.45:self.missing.setdefault(g,stamp)
            else:self.missing.pop(g,None)
            if stamp-self.missing.get(g,stamp)<=LOSS_SECONDS:usable.append(g)
        forward,status,_=cv2.calcOpticalFlowPyrLK(self.gray,gray,self.points,None,winSize=(21,21),maxLevel=2)
        if forward is None:
            self._anchor(gray,p,stamp);self.regional_base={};self.regional_epoch={}
            return dict(no,reason='flow_missing')
        backward,backstatus,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,forward,None,winSize=(21,21),maxLevel=2)
        if backward is None:
            self._anchor(gray,p,stamp);self.regional_base={};self.regional_epoch={}
            return dict(no,reason='reverse_flow_missing')
        good=(status[:,0]>0)&(backstatus[:,0]>0)&(np.linalg.norm(backward[:,0]-self.points[:,0],axis=1)<.7)
        delta=forward[:,0]-self.points[:,0];dist=np.linalg.norm(delta,axis=1)
        groups={};matched={}
        for g in self.required:
            selected=self.labels==g;valid=selected&good
            ok=g in usable and valid.sum()>=4 and valid.sum()>=selected.sum()*.65
            groups[str(g)]={'valid':bool(ok),'quiet':False}
            if ok:matched[g]=valid
        torso=good & np.isin(self.labels,[g for g in matched if g<2])
        translation=np.median(delta[torso],axis=0) if torso.any() else np.zeros(2)
        measurements=[];residuals=[]
        for g,valid in matched.items():
            travel=float(np.percentile(dist[valid],80))
            residual=float(np.percentile(np.linalg.norm(delta[valid]-translation,axis=1),80))
            position=self.regional_base.get(g,np.zeros(2))+np.median(delta[valid],axis=0)
            groups[str(g)].update(quiet=travel<=1.25,flow_px=travel,residual_px=residual,
                                 position_px=position.tolist(),position_epoch=self.regional_epoch.get(g,0))
            measurements.append(travel);residuals.append(residual)
        # Unknown regions are not votes for stillness; expose them for bounded
        # joint loss handling, while retaining other matched regions' evidence.
        enough=any(g<2 for g in matched) and sum(g>=2 for g in matched)>=2
        if not enough:
            self._anchor(gray,p,stamp)
            self.regional_base={};self.regional_epoch={}
            return dict(no,regions=groups,reason='insufficient_regions')
        maximum=max(measurements)
        coherent=max(residuals)<=.70 and np.linalg.norm(translation)<=12
        mode='translate' if coherent and np.linalg.norm(translation)>.65 else ('quiet' if maximum<=1.25 else 'moving')
        result=dict(valid=True,quiet=mode=='quiet',mode=mode,translation_px=translation.tolist(),stamp=stamp,
                    epoch=self.epoch,max_flow_px=maximum,regions=groups,region_count=len(matched),
                    covered=sorted({j for g in matched for j in SEGMENTS[g]}),reason=mode)
        if mode=='moving':
            # Preserve cumulative observed travel across a DIFFERENT limb's
            # re-anchor. Otherwise a moving arm could repeatedly reset the
            # torso's optical deadband and make a slow body slide get stuck.
            for g in range(len(SEGMENTS)):
                if g in matched:self.regional_base[g]=np.asarray(groups[str(g)]['position_px'])
                else:
                    self.regional_base.pop(g,None)
                    self.regional_epoch[g]=self.regional_epoch.get(g,0)+1
            self._anchor(gray,p,stamp)
        return result

class MaskStabilizer(OriginalMaskStabilizer):
    """Same current-evidence envelope; dense history flow only near foreground.

The 24-pixel guard exceeds the two-level flow window radius. Outside the
current/previous foreground union there is no history contribution to compute.
"""
    def apply(self,rgb,mask):
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        current=mask.astype(np.float32)/255
        self.flow_area=0
        if self.previous is not None and current.any():
            support=(current>.10)
            yy,xx=np.nonzero(support | (self.previous>.10))
            if len(xx):
                x0=max(0,int(xx.min())-24);x1=min(current.shape[1],int(xx.max())+25)
                y0=max(0,int(yy.min())-24);y1=min(current.shape[0],int(yy.max())+25)
                roi=np.s_[y0:y1,x0:x1];shape=current[roi].shape
                flow=cv2.calcOpticalFlowFarneback(gray[roi],self.gray[roi],None,.5,2,11,2,5,1.1,0)
                ry,rx=np.indices(shape,dtype=np.float32)
                # Sample the FULL history using global coordinates: a crop
                # must never clamp a hand at its internal ROI boundary.
                history=cv2.remap(self.previous,rx+x0+flow[:,:,0],ry+y0+flow[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                blended=.80*current[roi]+.20*history
                current[roi]=np.where(support[roi],np.maximum(current[roi]*.8,blended),0)
                self.flow_area=shape[0]*shape[1]
            current[~support]=0
        self.previous=current.copy();self.gray=gray
        return np.round(np.clip(current,0,1)*255).astype(np.uint8)
