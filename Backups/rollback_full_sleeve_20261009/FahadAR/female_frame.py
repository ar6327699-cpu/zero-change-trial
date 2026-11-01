"""Current camera framing and observed female arm targets, in 640x360 pixels."""
import numpy as np

def camera_frame(bgr,whole_sensor=False):
    import cv2
    h,w=bgr.shape[:2]
    if whole_sensor:
        scale=min(640/w,360/h);nw,nh=round(w*scale),round(h*scale)
        out=np.zeros((360,640,3),np.uint8)
        x,y=(640-nw)//2,(360-nh)//2
        out[y:y+nh,x:x+nw]=cv2.resize(bgr,(nw,nh))
        return out
    wanted=w*9/16
    if h>wanted:
        y=int((h-wanted)/2);bgr=bgr[y:y+int(wanted)]
    else:
        wanted=h*16/9;x=int((w-wanted)/2);bgr=bgr[:,x:x+int(wanted)]
    return cv2.resize(bgr,(640,360))

def arm_visible(p,j,threshold=.5):
    # Keep a confidently observed joint at the image boundary. A narrow
    # 3% guard permits a partially clipped wrist; it is not a missing-joint
    # extrapolation. Confidence is never modified.
    return p[j,3]>=threshold and np.all(p[j,:2]>-.03) and np.all(p[j,:2]<1.03)

def anchor_wrists(pose,hands):
    """Use the associated current palm's wrist root when pose drifts to cloth."""
    from occlusion import tracked_hand,torso_pixels
    p=np.asarray(pose,float).copy();changed=[]
    if p.shape!=(33,4) or not np.isfinite(p).all():return pose,changed
    torso=torso_pixels(p)
    for side,e,w in [('left',13,15),('right',14,16)]:
        if not arm_visible(p,w,.5) or not arm_visible(p,e,.5):continue
        hand=tracked_hand(p,hands,side,w,torso)
        if hand is None:continue
        length=np.linalg.norm(hand[0]-p[e,:2]*[640,360])
        if not 4<length<torso*1.05:continue
        if np.linalg.norm(hand[0]-p[w,:2]*[640,360])>1.:
            p[w,:2]=hand[0]/[640,360];changed.append(w)
    return p.tolist(),changed

def current_arm_targets(p,observed,motion,stamp):
    """Bound moving-arm fusion error against current high-confidence XY.

    Quiet regions retain their existing anchor. A valid moving image region
    plus a plausible current arm is required; unknown imagery cannot promote
    a detector jump. This is shared by both arms.
    """
    from female_reliability import fresh
    result=np.asarray(p,float).copy();raw=np.asarray(observed,float)
    corrected=[]
    if raw.shape!=(33,4) or not np.isfinite(raw).all() or not fresh(motion,stamp):return result,corrected
    regions=motion.get('regions',{})
    torso=np.mean([np.linalg.norm((raw[a,:2]-raw[b,:2])*[640,360]) for a,b in [(11,23),(12,24)]])
    for side,s,e,w,gs in [('left',11,13,15,(2,3)),('right',12,14,16,(4,5))]:
        evidence=[regions.get(str(g),{}) for g in gs]
        moving=side in motion.get('arm_follow_active',[]) or all(
            r.get('valid') and not r.get('quiet') and not r.get('rigid_translation') for r in evidence)
        if not moving or not all(arm_visible(raw,j,.70) for j in (s,e,w)):continue
        if np.linalg.norm((raw[s,:2]-result[s,:2])*[640,360])>3:continue
        lengths=[np.linalg.norm((raw[a,:2]-raw[b,:2])*[640,360]) for a,b in [(s,e),(e,w)]]
        if torso<20 or min(lengths)<4 or max(lengths)>torso*1.05:continue
        for j in (e,w):
            error=np.linalg.norm((raw[j,:2]-result[j,:2])*[640,360])
            if 1.5<error<=20:
                result[j,:2]=raw[j,:2];corrected.append(j)
    return result,corrected
