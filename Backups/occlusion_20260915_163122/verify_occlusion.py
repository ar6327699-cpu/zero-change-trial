from pathlib import Path
from types import SimpleNamespace as NS
import ast, importlib.util, json, sys, time, hashlib
import cv2, numpy as np

ROOT=Path('C:/Users/eisha/OneDrive/Desktop/AR Garment Studio')
TMP=Path(__file__).parent
sys.path.insert(0,str(ROOT/'FahadLive'))
cv2.setNumThreads(2)
def load(name,path):
    ast.parse(path.read_text(encoding='utf-8'))
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=load('old',TMP/'occlusion.py')
new=load('new',ROOT/'FahadAR/occlusion.py')
def marks(a):return [NS(x=x,y=y,z=z,visibility=v) for x,y,z,v in a]
def fixture(y=.5):
    p=np.zeros((33,4));p[:,:2]=.5;p[:,3]=1;p[17:23,3]=0
    p[11,:2]=[.65,.28];p[12,:2]=[.35,.28];p[23,:2]=[.6,.85];p[24,:2]=[.4,.85]
    p[13,:2]=[.68,.65];p[15,:2]=[.49,y];p[14,:2]=[.32,.65];p[16,:2]=[.65,.53]
    w=np.zeros((33,4));w[:,3]=1;w[[13,14,15,16],2]=-.12
    return p,w
def hand_at(p,k):
    # Palm extends upward from wrist; fingers have separate tips.
    offsets=np.array([[0,0],[-8,-5],[-15,-10],[-21,-15],[-25,-20],[-12,-17],[-13,-28],[-14,-36],[-15,-43],
        [-3,-20],[-3,-33],[-3,-43],[-3,-51],[6,-18],[7,-31],[8,-41],[9,-48],[14,-14],[17,-24],[20,-31],[23,-37]],float)
    h=np.zeros((21,4));h[:,3]=1;h[:,:2]=p[k,:2]+offsets/[640,360];return h
def result(p,w,h=None,face=None):
    return NS(pose_landmarks=marks(p),pose_world_landmarks=marks(w),face_landmarks=marks(face) if face is not None else [],
        left_hand_landmarks=marks(h) if h is not None else [],right_hand_landmarks=[])
def pixel(mask,point):
    x,y=np.round(point*[640,360]).astype(int);return int(mask[y,x])
probs={'skin':np.ones((360,640),np.float32),'head':np.zeros((360,640),np.float32)}
checks={};before={}
def check(name,fn,regression=False):
    try:checks[name]=bool(fn(new))
    except Exception as e:checks[name]=str(e)
    if regression:
        try:before[name]=bool(fn(old))
        except Exception as e:before[name]=str(e)
    print(name,checks[name],('before='+str(before[name])) if regression else '')

def hand_without_elbow(m,y):
    p,w=fixture(y);p[13,3]=.1;h=hand_at(p,15)
    mask,_=m.compose_matte(p,w,[],{'left':h},probs)
    return pixel(mask,h[9,:2])>230
check('hand_on_chest_missing_elbow',lambda m:hand_without_elbow(m,.5),True)
check('hand_on_stomach_missing_elbow',lambda m:hand_without_elbow(m,.75),True)
def crossed_depth(m):
    p,w=fixture();w[11,2]=-.23;w[23,2]=-.23;w[12,2]=.20;w[24,2]=.20
    w[[13,15],2]=-.09;h=hand_at(p,15)
    mask,_=m.compose_matte(p,w,[],{'left':h},probs)
    return pixel(mask,h[9,:2])>230
check('crossed_hand_uses_local_torso_depth',crossed_depth,True)
def profile(m,far=False):
    p,w=fixture();p[11,:2]=[.52,.28];p[23,:2]=[.52,.85];p[12,:2]=[.49,.28];p[24,:2]=[.49,.85]
    w[[11,23],2]=-.18;w[[12,24],2]=.18;w[[13,15],2]=.3 if far else -.10
    p[14,3]=0;p[16,3]=0;h=hand_at(p,15)
    mask,_=m.compose_matte(p,w,[],{'left':h},probs)
    return pixel(mask,h[9,:2])==0 if far else pixel(mask,h[9,:2])>230
check('profile_visible_hand_with_depth_noise',profile,True)
check('profile_rear_hand_stays_hidden',lambda m:profile(m,True))
def no_face(m):
    p,w=fixture();p[13,3]=.1;h=hand_at(p,15)
    rgb=np.full((360,640,3),[25,35,95],np.uint8)
    cv2.fillConvexPoly(rgb,cv2.convexHull(np.round(h[:,:2]*[640,360]).astype(np.int32)),(180,125,90))
    matte=m.FastForeground();matte.last_face=time.monotonic()-10
    mask,_=matte.process(rgb,result(p,w,h))
    return pixel(mask,h[9,:2])>230 and mask[260,340]==0
check('side_hand_works_without_face_reference',no_face,True)
def palm_colour(m):
    p,w=fixture();h=hand_at(p,15)
    rgb=np.full((360,640,3),[25,35,95],np.uint8)
    cv2.fillConvexPoly(rgb,cv2.convexHull(np.round(h[:,:2]*[640,360]).astype(np.int32)),(215,180,155))
    matte=m.FastForeground();matte.reference=(np.array([160.,92.]),np.array([5.,5.]));matte.last_face=time.monotonic()
    mask,_=matte.process(rgb,result(p,w,h))
    return pixel(mask,h[9,:2])>230
check('palm_lighting_differs_from_face',palm_colour,True)
def rear(m):
    p,w=fixture();w[[13,14,15,16],2]=.25
    mask,info=m.compose_matte(p,w,[],{},probs)
    return not mask.any() and info['arms']==0
check('both_rear_arms_excluded',rear)
def full(m):
    p,w=fixture();p[14,3]=0;p[16,3]=0;h=hand_at(p,15)
    mask,_=m.compose_matte(p,w,[],{'left':h},probs,True)
    middle=p[[13,15],:2].mean(0)
    return pixel(mask,middle)==0 and pixel(mask,h[9,:2])>230
check('full_sleeves_keep_forearm_covered',full)
def invalid(m):
    p,w=fixture();p[15,0]=np.nan
    return not m.compose_matte(p,w,[],{},probs)[0].any() and not m.compose_matte([],[],[],{},probs)[0].any()
check('missing_invalid_pose_clears_matte',invalid)
def no_skin(m):
    p,w=fixture();zero=np.zeros((360,640),np.float32)
    return not m.compose_matte(p,w,[],{},dict(skin=zero,head=zero))[0].any()
check('non_skin_pixels_excluded',no_skin)
def mismatched(m):
    p,w=fixture();p[13,3]=0;h=hand_at(p,15);h[:,0]+=.35
    mask,_=m.compose_matte(p,w,[],{'left':h},probs)
    return pixel(mask,h[9,:2])==0
check('unrelated_hand_landmarks_rejected',mismatched)
def reset(m):
    p,w=fixture();f=m.FastForeground();f.reference=(np.array([155.,100.]),np.array([8.,8.]));f.last_face=time.monotonic()
    f.reset();mask,_=f.process(np.full((360,640,3),[180,125,90],np.uint8),result(p,w))
    return not mask.any()
check('new_person_reset_drops_old_skin_reference',reset)
def old_front(m):
    p,w=fixture();p[15,:2]=[.4,.45];p[16,:2]=[.6,.45]
    mask,info=m.compose_matte(p,w,[],{},probs)
    return info['arms']==2 and mask[185,320]>200 and mask[300,320]==0
check('original_crossed_forearm_coverage',old_front)
def fingers(m):
    p,w=fixture();p[13,3]=.1
    for i,x in [(17,.47),(19,.50),(21,.53)]:p[i,:2]=[x,.39];p[i,3]=1
    mask,_=m.compose_matte(p,w,[],{},probs)
    return pixel(mask,np.array([.5,.415]))>200
check('pose_fingers_when_detailed_hand_missing',fingers,True)

# Existing calibration remains intact; both 90-degree profiles retain scale
# with only the near shoulder/hip visible.
cal=load('cal',ROOT/'FahadAR/calibration.py')
p,w=fixture();src=np.array([[.1,1.],[-.1,1.],[.08,0.],[-.08,0.]])
for hidden in ([11,23],[12,24]):
    q=p.copy();q[hidden,3]=.1;q[[11,12],0]=.5;q[[23,24],0]=.5
    fit=cal.locked_fit(src,q,1.)
    checks['profile_fit_near_pair_'+str(hidden[0])]=bool(fit is not None and np.isfinite(fit[0]).all() and fit[0][0]>.1)
for rel in ['FahadAR/calibration.py','FahadLive/fahad_backend.py']:
    checks['known_good_'+rel]=(ROOT/rel).read_bytes()==(ROOT.with_name('AR Baised')/rel).read_bytes()

p,w=fixture();h=hand_at(p,15);rgb=np.full((360,640,3),[180,125,90],np.uint8);r=result(p,w,h)
bench={}
for label,module in [('before',old),('after',new)]:
    f=module.FastForeground();f.reference=(np.array([153.,103.]),np.array([8.,8.]));f.last_face=time.monotonic()
    times=[]
    for i in range(24):
        t=time.perf_counter();f.process(rgb,r);times.append((time.perf_counter()-t)*1000)
    bench[label]=round(float(np.median(times[4:])),2)
report={'checks':checks,'before_fix_regressions':before,'smooth_median_ms':bench,
        'scope':'Synthetic poses/colours and saved solver/calibration checks. Live wearer edge accuracy still needs verification.'}
(TMP/'ar_studio_occlusion_test_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
assert all(v is True for v in checks.values()), 'Regression check failed'
