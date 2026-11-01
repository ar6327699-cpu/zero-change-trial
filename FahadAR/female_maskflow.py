"""Full-resolution matte history flow in disjoint foreground neighbourhoods."""
import cv2
import numpy as np
from female_image_reliability import MaskStabilizer as PreviousMask

class MaskStabilizer(PreviousMask):
    @staticmethod
    def boxes(support):
        count,_,stats,_=cv2.connectedComponentsWithStats(support.astype(np.uint8),8)
        h,w=support.shape
        boxes=[]
        for x,y,sw,sh,_ in stats[1:]:
            x,y,sw,sh=map(int,(x,y,sw,sh))
            box=[max(0,x-24),max(0,y-24),min(w,x+sw+24),min(h,y+sh+24)]
            # Merge overlapping guards to avoid two incompatible flow fields
            # within one skin component. Repeat for transitive overlaps.
            i=0
            while i<len(boxes):
                a=boxes[i]
                if box[0]<=a[2] and a[0]<=box[2] and box[1]<=a[3] and a[1]<=box[3]:
                    box=[min(box[0],a[0]),min(box[1],a[1]),max(box[2],a[2]),max(box[3],a[3])]
                    boxes.pop(i);i=0
                else:i+=1
            boxes.append(box)
        return boxes

    def apply(self,rgb,mask):
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        current=mask.astype(np.float32)/255
        self.flow_area=0
        if self.previous is not None and current.any():
            support=current>.10
            for x0,y0,x1,y1 in self.boxes(support|(self.previous>.10)):
                roi=np.s_[y0:y1,x0:x1];shape=current[roi].shape
                flow=cv2.calcOpticalFlowFarneback(gray[roi],self.gray[roi],None,.5,2,11,2,5,1.1,0)
                ry,rx=np.indices(shape,dtype=np.float32)
                history=cv2.remap(self.previous,rx+x0+flow[:,:,0],ry+y0+flow[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                blended=.80*current[roi]+.20*history
                current[roi]=np.where(support[roi],np.maximum(current[roi]*.8,blended),0)
                self.flow_area+=shape[0]*shape[1]
            current[~support]=0
        self.previous=current.copy();self.gray=gray
        return np.round(np.clip(current,0,1)*255).astype(np.uint8)
