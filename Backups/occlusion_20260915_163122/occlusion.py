"""Same-frame semantic skin matte constrained by pose and front/back ordering.

This is an exposed-hand/forearm matte, not generic cloth simulation or a depth
camera. No masks/images are persisted. Missing evidence produces no old matte.
"""
from pathlib import Path
import time
import cv2
import numpy as np

W,H=640,360

def compose_matte(pose,world,face,hands,probabilities,full_sleeves=False):
    out=np.zeros((H,W),np.uint8);roi=np.zeros_like(out)
    p=np.asarray(pose,float);w=np.asarray(world,float)
    if p.shape!=(33,4) or w.shape!=(33,4) or not np.isfinite(p).all() or not np.isfinite(w).all():
        return out,{'arms':0,'head':False}
    xy=p[:,:2]*[W,H]
    def valid(i):return p[i,3]>.45 and -.05<p[i,0]<1.05 and -.05<p[i,1]<1.05
    def pt(v):return tuple(np.round(np.clip(v,[-W,-H],[2*W,2*H])).astype(int))
    torso=max(np.linalg.norm((xy[11]+xy[12]-xy[23]-xy[24])*.5),30.)
    radius=int(np.clip(torso*.095,4,24));count=0
    for s,e,k,side in [(11,13,15,'left'),(12,14,16,'right')]:
        if not (valid(s) and valid(e) and valid(k)):continue
        # MediaPipe world z decreases toward camera. Comparing with the same
        # shoulder avoids resurrecting the far arm through a turned torso.
        front=w[k,2]<=w[s,2]+.045 or w[e,2]<w[s,2]-.025
        if not front:continue
        count+=1
        if not full_sleeves:
            cv2.line(roi,pt(xy[e]),pt(xy[k]),255,2*radius,cv2.LINE_AA)
            cv2.circle(roi,pt(xy[e]),radius,255,-1,cv2.LINE_AA)
        cv2.circle(roi,pt(xy[k]),max(2,int(radius*(.55 if full_sleeves else 1.25))),255,-1,cv2.LINE_AA)
        hand=np.asarray(hands.get(side,[]),float)
        if hand.shape==(21,4) and np.isfinite(hand).all():
            hxy=hand[:,:2]*[W,H]
            # Holistic handedness mismatches or stale detector hands must not
            # open holes at unrelated positions on the shirt.
            if np.linalg.norm(hxy[0]-xy[k])<torso*.4:
                hull=cv2.convexHull(np.array([pt(v) for v in hxy],np.int32))
                handroi=np.zeros_like(roi);cv2.fillConvexPoly(handroi,hull,255)
                handroi=cv2.dilate(handroi,np.ones((5,5),np.uint8))
                roi=np.maximum(roi,handroi)
    head_roi=np.zeros_like(out);f=np.asarray(face,float)
    if f.ndim==2 and len(f)>=468 and f.shape[1]>=2 and np.isfinite(f).all():
        fx=f[:,:2]*[W,H]
        # Face + a narrow neck bridge; never classify an open shirt's exposed
        # chest as a foreground hand. Neck bridge ends at shoulder line.
        lo=fx.min(0);hi=fx.max(0);centre=(lo+hi)*.5
        cv2.ellipse(head_roi,pt(centre),tuple(np.maximum((hi-lo)*[.64,.72],2).astype(int)),0,0,360,255,-1)
        if valid(11) and valid(12):
            sh=(xy[11]+xy[12])*.5;chin=fx[152]
            half=max(4,(hi[0]-lo[0])*.28)
            poly=np.array([pt(chin+[-half,-3]),pt(chin+[half,-3]),pt(sh+[half,-torso*.015]),pt(sh+[-half,-torso*.015])])
            cv2.fillConvexPoly(head_roi,cv2.convexHull(poly),255)
    skin=np.clip((probabilities['skin']-.30)/.40,0,1)
    head=np.clip((probabilities['head']-.30)/.40,0,1)
    # Stay in float32: uint8 / 255 previously promoted the whole frame to
    # float64 and allocated three full-size compositing intermediates.
    arm_weight=roi.astype(np.float32)*(1./255.)
    head_weight=head_roi.astype(np.float32)*(1./255.)
    alpha=np.maximum(skin*arm_weight,np.maximum(skin,head)*head_weight)
    alpha=cv2.GaussianBlur(alpha,(3,3),.55)
    return np.round(np.clip(alpha,0,1)*255).astype(np.uint8),{'arms':count,'head':bool(head_roi.any())}

class Foreground:
    def __init__(self,full_sleeves=False):
        self.full_sleeves=full_sleeves
        import mediapipe as mp
        self.mp=mp
        model=Path(__file__).with_name('models')/'selfie_multiclass_256x256.tflite'
        if not model.is_file():raise RuntimeError('Local skin segmentation model is missing.')
        opt=mp.tasks.vision.ImageSegmenterOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,output_category_mask=False,output_confidence_masks=True)
        self.model=mp.tasks.vision.ImageSegmenter.create_from_options(opt)
    def infer(self,rgb,previous_pose,stamp):
        # Previous pose selects only a generous crop. Segmentation itself and
        # the compositing pose are always from THIS frame, never stale masks.
        x0,y0,x1,y1=0,0,W,H
        a=np.asarray(previous_pose,float)
        if a.shape==(33,4) and np.isfinite(a).all():
            valid=(a[:,3]>.45)&(a[:,0]>0)&(a[:,0]<1)&(a[:,1]>0)&(a[:,1]<1)
            ids=[i for i in range(25) if valid[i]]
            if len(ids)>=4:
                xy=a[ids,:2]*[W,H];lo=xy.min(0);hi=xy.max(0);pad=np.maximum((hi-lo)*.30,[40,40])
                lo=np.maximum(lo-pad,0).astype(int);hi=np.minimum(hi+pad,[W,H]).astype(int)
                if np.min(hi-lo)>20:x0,y0=lo;x1,y1=hi
        crop=np.ascontiguousarray(rgb[y0:y1,x0:x1])
        seg=self.model.segment_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=crop),stamp)
        masks=seg.confidence_masks
        if len(masks)!=6:raise RuntimeError('Unexpected skin model label count')
        skin=np.zeros((H,W),np.float32);head=skin.copy()
        # Official order: background, hair, body-skin, face-skin, clothes, other.
        skin[y0:y1,x0:x1]=cv2.resize(masks[2].numpy_view()+masks[3].numpy_view(),(x1-x0,y1-y0))
        head[y0:y1,x0:x1]=cv2.resize(masks[1].numpy_view()+masks[3].numpy_view(),(x1-x0,y1-y0))
        return {'skin':skin,'head':head}
    def compose(self,result,probabilities):
        from fahad_backend import points
        return compose_matte(points(result.pose_landmarks),points(result.pose_world_landmarks),points(result.face_landmarks),
            {'left':points(result.left_hand_landmarks),'right':points(result.right_hand_landmarks)},probabilities,self.full_sleeves)
    def close(self):self.model.close()

class FastForeground:
    """Low-cost adaptive skin chroma + current-frame pose.

    Deliberate speed/quality tradeoff: not semantic segmentation. Skin-coloured
    clothes or dramatic light changes can confuse it; Quality mode is retained.
    """
    def __init__(self,full_sleeves=False):self.full_sleeves=full_sleeves;self.reset()
    def reset(self):self.reference=None;self.last_face=-1e9
    def process(self,rgb,result):
        from fahad_backend import points
        face=points(result.face_landmarks);f=np.asarray(face,float)
        # Estimate broad skin colour at half resolution; the geometric arm/
        # hand gate below remains full resolution. Avoid full-frame float
        # chroma arithmetic on every 640x360 pixel on this CPU.
        small=cv2.resize(rgb,(W//2,H//2),interpolation=cv2.INTER_AREA)
        chroma=cv2.cvtColor(small,cv2.COLOR_RGB2YCrCb).astype(np.float32)
        fh,fw=chroma.shape[:2]
        if len(f)>=468 and np.isfinite(f).all():
            lo=f[:,:2].min(0)*[fw,fh];hi=f[:,:2].max(0)*[fw,fh];size=hi-lo
            samples=[]
            if min(size)>9:
                for xa,xb,ya,yb in [(.20,.36,.32,.52),(.64,.80,.32,.52),(.43,.57,.42,.60)]:
                    x0,y0=np.maximum(lo+size*[xa,ya],0).astype(int);x1,y1=np.minimum(lo+size*[xb,yb],[fw,fh]).astype(int)
                    patch=chroma[y0:y1,x0:x1].reshape(-1,3)
                    patch=patch[(patch[:,0]>20)&(patch[:,0]<250)]
                    if len(patch):samples.append(patch[:,1:])
                if samples:
                    samples=np.concatenate(samples);centre=np.median(samples,axis=0)
                    spread=np.clip(np.median(np.abs(samples-centre),axis=0)*1.4826,5,12)
                    self.reference=(centre,spread);self.last_face=time.monotonic()
        if self.reference is None or time.monotonic()-self.last_face>2:
            return np.zeros((H,W),np.uint8),{'arms':0,'head':False,'mode':'smooth: face needed'}
        centre,spread=self.reference;d=np.sum(((chroma[:,:,1:]-centre)/spread)**2,axis=2)
        confidence=cv2.resize(np.clip((8-d)/5,0,1),(W,H),interpolation=cv2.INTER_LINEAR)
        mask,info=compose_matte(points(result.pose_landmarks),points(result.pose_world_landmarks),face,
            {'left':points(result.left_hand_landmarks),'right':points(result.right_hand_landmarks)},
            {'skin':confidence,'head':confidence},self.full_sleeves)
        info['mode']='smooth adaptive skin';return mask,info
    def close(self):pass
