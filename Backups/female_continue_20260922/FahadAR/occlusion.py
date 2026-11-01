"""Current-frame exposed hands/forearms, with local torso depth ordering.

Smooth mode uses face and visible palm colour; Semantic uses the local model.
No camera images or wearer references are saved. Missing evidence clears masks.
"""
from pathlib import Path
import time
import cv2
import numpy as np

W,H=640,360


def tracked_hand(p,hands,side,wrist,torso):
    """Associate a current hand with its pose wrist, independently of the elbow."""
    hand=np.asarray(hands.get(side,[]),float)
    if hand.shape!=(21,4) or not np.isfinite(hand).all():return None
    if p[wrist,3]<=.2 or not (-.05<p[wrist,0]<1.05 and -.05<p[wrist,1]<1.05):return None
    xy=hand[:,:2]*[W,H]
    if np.linalg.norm(xy[0]-p[wrist,:2]*[W,H])>max(12.,torso*.30):return None
    size=np.ptp(xy,axis=0)
    palm=np.linalg.norm(xy[5]-xy[17])
    if max(size)>torso*.9 or not 2.<palm<torso*.5:return None
    if not np.all((hand[[0,5,9,13,17],:2]>-.05)&(hand[[0,5,9,13,17],:2]<1.05)):return None
    return xy


def torso_pixels(p):
    lengths=[np.linalg.norm((p[a,:2]-p[b,:2])*[W,H])
             for a,b in [(11,23),(12,24)] if min(p[a,3],p[b,3])>.45]
    return max(float(np.median(lengths)),30.) if lengths else 90.


def visible_arm_regions(p,hands,skin,full_sleeves=False):
    """Recover observed skin crossing the shirt despite noisy pose depth.

    Only small skin components connected to current arm/hand landmarks may
    override depth. A large skin-coloured torso/background is not an arm.
    Recovery is confined to the torso overlap; the other matte is unchanged.
    """
    torso=torso_pixels(p);radius=int(np.clip(torso*.095,4,24))
    xy=p[:,:2]*[W,H];out=np.zeros((H,W),np.uint8)
    def seen(i,threshold=.2):return p[i,3]>threshold and 0<p[i,0]<1 and 0<p[i,1]<1
    def pt(v):return tuple(np.round(np.clip(v,[0,0],[W-1,H-1])).astype(int))
    ids=[i for i in [11,12,24,23] if seen(i)]
    if len(ids)<3:return out
    torso_roi=np.zeros_like(out)
    cv2.fillConvexPoly(torso_roi,cv2.convexHull(np.array([pt(xy[i]) for i in ids],np.int32)),255)
    torso_roi=cv2.dilate(torso_roi,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
    binary=(skin>.55).astype(np.uint8)
    binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    _,labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
    for side,e,k in [('left',13,15),('right',14,16)]:
        if not seen(k):continue
        hand=tracked_hand(p,hands,side,k,torso)
        if hand is None and not seen(e,.45):continue
        corridor=np.zeros_like(out);seed=np.zeros_like(out)
        if not full_sleeves and seen(e,.45):
            cv2.line(corridor,pt(xy[e]),pt(xy[k]),255,2*radius)
            cv2.circle(seed,pt(xy[e]),max(2,radius//2),255,-1)
        cv2.circle(corridor,pt(xy[k]),max(2,int(radius*(.55 if full_sleeves else 1.25))),255,-1)
        cv2.circle(seed,pt(xy[k]),max(2,radius//2),255,-1)
        if hand is not None:
            hull=cv2.convexHull(np.array([pt(v) for v in hand],np.int32))
            cv2.fillConvexPoly(corridor,hull,255);cv2.fillConvexPoly(seed,hull,255)
        else:
            tips=[xy[i] for i in [k+2,k+4,k+6] if seen(i) and np.linalg.norm(xy[i]-xy[k])<torso*.35]
            if len(tips)>=2:
                hull=cv2.convexHull(np.array([pt(xy[k]),*[pt(v) for v in tips]],np.int32))
                cv2.fillConvexPoly(corridor,hull,255)
        seed_labels,seed_counts=np.unique(labels[seed>0],return_counts=True)
        for label,n in zip(seed_labels,seed_counts):
            if label==0 or n<3:continue
            area=stats[label,cv2.CC_STAT_AREA]
            if not 4<=area<=max(40,torso*torso*.8):continue
            component=labels==label
            inside=component&(corridor>0)
            if np.count_nonzero(inside)<max(4,area*.35):continue
            out[inside&(torso_roi>0)]=255
    return out


def compose_matte(pose,world,face,hands,probabilities,full_sleeves=False):
    out=np.zeros((H,W),np.uint8);roi=np.zeros_like(out)
    p=np.asarray(pose,float);w=np.asarray(world,float)
    if p.shape!=(33,4) or w.shape!=(33,4) or not np.isfinite(p).all() or not np.isfinite(w).all():
        return out,{'arms':0,'head':False}
    xy=p[:,:2]*[W,H]
    def valid(i):return p[i,3]>.45 and -.05<p[i,0]<1.05 and -.05<p[i,1]<1.05
    def pt(v):return tuple(np.round(np.clip(v,[-W,-H],[2*W,2*H])).astype(int))
    torso=torso_pixels(p)
    radius=int(np.clip(torso*.095,4,24));count=0

    def torso_depth(point):
        # Interpolate the torso at THIS screen position. A crossed wrist is
        # not necessarily in front of its own shoulder, especially in profile.
        samples=[]
        for shoulder,hip in [(11,23),(12,24)]:
            if not valid(shoulder):continue
            t=np.clip((point[1]-xy[shoulder,1])/max(xy[hip,1]-xy[shoulder,1],1),0,1) if valid(hip) else 0.
            samples.append(((1-t)*xy[shoulder,0]+t*xy[hip,0],(1-t)*w[shoulder,2]+t*w[hip,2]))
        if not samples:return None
        if len(samples)==1:return samples[0][1]
        samples.sort()
        (x0,z0),(x1,z1)=samples
        if x1-x0<torso*.12:return min(z0,z1)
        t=np.clip((point[0]-x0)/(x1-x0),0,1)
        return (1-t)*z0+t*z1

    for s,e,k,side in [(11,13,15,'left'),(12,14,16,'right')]:
        hand=tracked_hand(p,hands,side,k,torso)
        if not valid(k) and hand is None:continue
        wrist_depth=torso_depth(xy[k])
        if wrist_depth is None:continue
        # Reliable visible hand landmarks allow modest monocular-depth noise,
        # while clearly rear hands remain excluded. No previous-frame hulls.
        wrist_front=w[k,2]<=wrist_depth+(.12 if hand is not None else .065)
        armroi=np.zeros_like(roi)
        if not full_sleeves and valid(e):
            elbow_depth=torso_depth(xy[e])
            d0=w[e,2]-elbow_depth;d1=w[k,2]-wrist_depth
            # Clip a forearm at the torso depth crossing; an elbow in front
            # must not pull a rear wrist (or vice versa) through the garment.
            tolerance=.065
            if min(d0,d1)<=tolerance:
                a,b=xy[e].copy(),xy[k].copy()
                if max(d0,d1)>tolerance:
                    cross=a+(b-a)*np.clip((tolerance-d0)/(d1-d0),0,1)
                    if d0>tolerance:a=cross
                    else:b=cross
                cv2.line(armroi,pt(a),pt(b),255,2*radius,cv2.LINE_AA)
        if wrist_front:
            cv2.circle(armroi,pt(xy[k]),max(2,int(radius*(.55 if full_sleeves else 1.25))),255,-1,cv2.LINE_AA)
            if hand is not None:
                hull=cv2.convexHull(np.array([pt(v) for v in hand],np.int32))
                handroi=np.zeros_like(roi);cv2.fillConvexPoly(handroi,hull,255)
                handroi=cv2.dilate(handroi,np.ones((5,5),np.uint8))
                armroi=np.maximum(armroi,handroi)
            else:
                # Pose also tracks thumb/index/pinky when Holistic loses the
                # detailed hand. Restrict this fallback to nearby seen digits.
                fingers=[k+2,k+4,k+6]
                tips=[xy[i] for i in fingers if valid(i) and np.linalg.norm(xy[i]-xy[k])<torso*.35]
                if len(tips)>=2:
                    hull=cv2.convexHull(np.array([pt(xy[k]),*[pt(v) for v in tips]],np.int32))
                    cv2.fillConvexPoly(armroi,hull,255)
        if armroi.any():count+=1;roi=np.maximum(roi,armroi)
    head_roi=np.zeros_like(out);f=np.asarray(face,float)
    if f.ndim==2 and len(f)>=468 and f.shape[1]>=2 and np.isfinite(f).all():
        fx=f[:,:2]*[W,H]
        lo=fx.min(0);hi=fx.max(0);centre=(lo+hi)*.5
        cv2.ellipse(head_roi,pt(centre),tuple(np.maximum((hi-lo)*[.64,.72],2).astype(int)),0,0,360,255,-1)
        if valid(11) and valid(12):
            sh=(xy[11]+xy[12])*.5;chin=fx[152]
            half=max(4,(hi[0]-lo[0])*.28)
            poly=np.array([pt(chin+[-half,-3]),pt(chin+[half,-3]),pt(sh+[half,-torso*.015]),pt(sh+[-half,-torso*.015])])
            cv2.fillConvexPoly(head_roi,cv2.convexHull(poly),255)
    skin=np.clip((probabilities['skin']-.30)/.40,0,1)
    head=np.clip((probabilities['head']-.30)/.40,0,1)
    roi=np.maximum(roi,visible_arm_regions(p,hands,skin,full_sleeves))
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
        # Previous pose only selects the crop; all probabilities are current.
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
        skin[y0:y1,x0:x1]=cv2.resize(masks[2].numpy_view()+masks[3].numpy_view(),(x1-x0,y1-y0))
        head[y0:y1,x0:x1]=cv2.resize(masks[1].numpy_view()+masks[3].numpy_view(),(x1-x0,y1-y0))
        return {'skin':skin,'head':head}
    def compose(self,result,probabilities):
        from fahad_backend import points
        return compose_matte(points(result.pose_landmarks),points(result.pose_world_landmarks),points(result.face_landmarks),
            {'left':points(result.left_hand_landmarks),'right':points(result.right_hand_landmarks)},probabilities,self.full_sleeves)
    def close(self):self.model.close()


class FastForeground:
    """Adaptive face/palm chroma + current pose, with no stored hand masks.

    Colour-based mode can confuse skin-coloured clothes; Semantic remains
    available for scenes that require neural skin classification.
    """
    def __init__(self,full_sleeves=False):self.full_sleeves=full_sleeves;self.reset()
    def reset(self):self.reference=None;self.last_face=-1e9
    def process(self,rgb,result):
        from fahad_backend import points
        face=points(result.face_landmarks);f=np.asarray(face,float)
        pose=points(result.pose_landmarks);p=np.asarray(pose,float)
        world=points(result.pose_world_landmarks)
        hands={'left':points(result.left_hand_landmarks),'right':points(result.right_hand_landmarks)}
        if p.shape!=(33,4) or not np.isfinite(p).all():
            self.reset()
            return np.zeros((H,W),np.uint8),{'arms':0,'head':False,'mode':'smooth: pose needed'}
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
        references=[]
        if self.reference is not None and time.monotonic()-self.last_face<=2:
            references.append(self.reference)
        # Palms may differ from the face in colour/lighting. Sample the inset
        # CURRENT palm polygon so a side turn does not switch off visible hands.
        for side,k in [('left',15),('right',16)]:
            hand=tracked_hand(p,hands,side,k,torso_pixels(p))
            if hand is None:continue
            palm=np.round(hand[[0,5,9,13,17]]*.5).astype(np.int32)
            sample_mask=np.zeros((fh,fw),np.uint8)
            cv2.fillConvexPoly(sample_mask,cv2.convexHull(palm),255)
            sample_mask=cv2.erode(sample_mask,np.ones((3,3),np.uint8))
            samples=chroma[sample_mask>0]
            samples=samples[(samples[:,0]>20)&(samples[:,0]<250)]
            if len(samples)<6:continue
            centre=np.median(samples[:,1:],axis=0)
            # Reject neutral/strongly saturated clothing samples. This broad
            # chroma guard is a Smooth-mode heuristic, not semantic labelling.
            cr,cb=centre
            if not (122<cr<190 and 60<cb<150 and cr-cb>8):continue
            spread=np.clip(np.median(np.abs(samples[:,1:]-centre),axis=0)*1.4826,6,14)
            references.append((centre,spread))
        if not references:
            return np.zeros((H,W),np.uint8),{'arms':0,'head':False,'mode':'smooth: face or hand needed'}
        confidence=np.zeros((fh,fw),np.float32)
        for centre,spread in references:
            d=np.sum(((chroma[:,:,1:]-centre)/spread)**2,axis=2)
            confidence=np.maximum(confidence,np.clip((8-d)/5,0,1))
        confidence=cv2.resize(confidence,(W,H),interpolation=cv2.INTER_LINEAR)
        mask,info=compose_matte(pose,world,face,hands,{'skin':confidence,'head':confidence},self.full_sleeves)
        info['mode']='smooth adaptive face/palm skin';return mask,info
    def close(self):pass
