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


