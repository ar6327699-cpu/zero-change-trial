"""Fill spatial gaps in weak arm feature sets without lowering evidence gates."""
import cv2
import numpy as np
from female_image_reliability import ImageMotion as ExistingSelector
from female_motion import SEGMENTS

class PatchSelector(ExistingSelector):
    def _anchor(self,gray,p,stamp,groups=None):
        super()._anchor(gray,p,stamp,groups)
        xy=p[:,:2]*[640,360]
        for group in (2,3,4,5):
            if groups is not None and group not in groups:continue
            selected=self.labels==group
            if np.count_nonzero(selected)<4:continue
            a,b=SEGMENTS[group];axis=xy[b]-xy[a];length=np.linalg.norm(axis)
            if length<9:continue
            axis/=length;original=self.points[selected,0]
            if np.ptp(original@axis)>=length*.45:continue
            # Request corners separately along the SAME bounded inner sleeve
            # strip. Global strongest corners can all cluster near one cuff.
            radius=int(np.clip(length*.07,2,7))
            lo=np.maximum(np.floor(np.minimum(xy[a],xy[b])-radius-2),0).astype(int)
            hi=np.minimum(np.ceil(np.maximum(xy[a],xy[b])+radius+3),[640,360]).astype(int)
            x0,y0=lo;x1,y1=hi;crop=gray[y0:y1,x0:x1]
            points=[v.copy() for v in original]
            for start,end in ((.12,.35),(.35,.65),(.65,.88)):
                mask=np.zeros(crop.shape,np.uint8)
                u=xy[a]+(xy[b]-xy[a])*start-lo;v=xy[a]+(xy[b]-xy[a])*end-lo
                cv2.line(mask,tuple(u.astype(int)),tuple(v.astype(int)),255,2*radius)
                extra=cv2.goodFeaturesToTrack(crop,6,.025,2,mask=mask,blockSize=3)
                if extra is None:continue
                for point in extra[:,0]+lo:
                    if min(np.linalg.norm(point-q) for q in points)>=2:points.append(point)
            candidate=np.asarray(points,np.float32)
            if len(candidate)==len(original) or np.ptp(candidate@axis)<length*.4:continue
            self.points=np.concatenate([self.points[~selected],candidate[:,None,:]])
            self.labels=np.concatenate([self.labels[~selected],np.full(len(candidate),group,dtype=int)])
