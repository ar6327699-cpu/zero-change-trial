"""Female-only fixed-image motion evidence, before pose smoothing.

Tracks textured patches INSIDE observed torso/limb capsules, never the scene
background as a global stabilizer. The image anchor does not creep. This is
corroboration for the final-frame gate, not pose prediction or video freezing.
"""
import cv2
import numpy as np

SEGMENTS=((11,23),(12,24),(11,13),(13,15),(12,14),(14,16),
          (23,25),(25,27),(24,26),(26,28))

class ImageMotion:
    def __init__(self): self.reset()
    def reset(self):
        self.gray=None; self.points=None; self.labels=None; self.last=None
        self.required=(); self.epoch=0

    def _anchor(self,gray,p,stamp):
        points=[]; labels=[]; required=[]
        xy=p[:,:2]*[640,360]
        for group,(a,b) in enumerate(SEGMENTS):
            if min(p[a,3],p[b,3])<.6: continue
            if not np.all((xy[[a,b]]>[4,4]) & (xy[[a,b]]<[636,356])): continue
            length=np.linalg.norm(xy[a]-xy[b])
            if length<9: continue
            # Narrow, inset strip avoids floor/wall corners at the silhouette.
            radius=int(np.clip(length*(.10 if group<2 else .07),2,7))
            mask=np.zeros(gray.shape,np.uint8)
            u=xy[a]*.8+xy[b]*.2; v=xy[a]*.2+xy[b]*.8
            cv2.line(mask,tuple(u.astype(int)),tuple(v.astype(int)),255,2*radius)
            low=np.maximum(np.floor(np.minimum(u,v)-radius-2).astype(int),0)
            high=np.minimum(np.ceil(np.maximum(u,v)+radius+3).astype(int),[640,360])
            x0,y0=low;x1,y1=high
            found=cv2.goodFeaturesToTrack(gray[y0:y1,x0:x1],18,.025,3,mask=mask[y0:y1,x0:x1],blockSize=3)
            if found is None or len(found)<4: continue
            found[:,0,:]+=low
            points.extend(found[:,0,:]);labels.extend([group]*len(found));required.append(group)
        self.gray=gray.copy();self.points=np.asarray(points,np.float32).reshape(-1,1,2)
        self.labels=np.asarray(labels);self.required=tuple(required);self.last=stamp;self.epoch+=1

    def update(self,rgb,pose,stamp):
        stamp=float(stamp);p=np.asarray(pose,float)
        no=dict(valid=False,quiet=False,stamp=stamp,epoch=self.epoch)
        if p.shape!=(33,4) or not np.isfinite(p).all() or not np.isfinite(stamp):
            self.reset();return no
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        if gray.shape!=(360,640):self.reset();return no
        if self.last is None or not 0<stamp-self.last<=.6 or len(self.points)<12:
            self._anchor(gray,p,stamp);return no
        self.last=stamp
        usable=[g for g in self.required if min(p[list(SEGMENTS[g]),3])>=.45]
        if len(usable)!=len(self.required) or not any(g<2 for g in usable) or sum(g>=2 for g in usable)<2:
            self._anchor(gray,p,stamp);return no
        forward,status,_=cv2.calcOpticalFlowPyrLK(self.gray,gray,self.points,None,winSize=(21,21),maxLevel=2)
        if forward is None:self._anchor(gray,p,stamp);return no
        backward,backstatus,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,forward,None,winSize=(21,21),maxLevel=2)
        if backward is None:self._anchor(gray,p,stamp);return no
        good=(status[:,0]>0)&(backstatus[:,0]>0)&(np.linalg.norm(backward[:,0]-self.points[:,0],axis=1)<.7)
        delta=forward[:,0]-self.points[:,0];dist=np.linalg.norm(delta,axis=1)
        measurements=[];residuals=[]
        torso=good & np.isin(self.labels,[0,1])
        translation=np.median(delta[torso],axis=0) if torso.any() else np.zeros(2)
        for group in usable:
            selected=self.labels==group;valid=selected&good
            if valid.sum()<4 or valid.sum()<selected.sum()*.65:
                self._anchor(gray,p,stamp);return no
            # A moving wrist/arm cannot be voted away by a stationary torso.
            measurements.append(float(np.percentile(dist[valid],80)))
            residuals.append(float(np.percentile(np.linalg.norm(delta[valid]-translation,axis=1),80)))
        maximum=max(measurements,default=999.)
        coherent=max(residuals,default=999.)<=.70 and np.linalg.norm(translation)<=12.
        mode='translate' if coherent and np.linalg.norm(translation)>.65 else ('quiet' if maximum<=1.25 else 'moving')
        quiet=mode=='quiet'
        result=dict(valid=True,quiet=quiet,mode=mode,translation_px=translation.tolist(),stamp=stamp,epoch=self.epoch,max_flow_px=maximum,regions=len(usable),
                    covered=sorted({j for g in usable for j in SEGMENTS[g]}))
        if mode=='moving':self._anchor(gray,p,stamp)
        return result
