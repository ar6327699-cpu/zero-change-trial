from pathlib import Path
from types import SimpleNamespace as NS
import ast,base64,cv2,numpy as np,json,sys,importlib.util,time
ROOT=Path('C:/Users/eisha/OneDrive/Desktop/AR Garment Studio');TMP=Path(__file__).parent
sys.path.insert(0,str(ROOT/'FahadLive'));cv2.setNumThreads(2)
source=(ROOT/'FahadAR/occlusion.py').read_text()
fragment=(TMP/'ar_hands_recovery_fragment.py').read_text()
assert 'def visible_arm_regions' not in source
candidate=source.replace('def compose_matte(',fragment+'def compose_matte(',1)
needle="    arm_weight=roi.astype(np.float32)*(1./255.)"
assert candidate.count(needle)==1
candidate=candidate.replace(needle,"    roi=np.maximum(roi,visible_arm_regions(p,hands,skin,full_sleeves))\n"+needle)
path=TMP/'ar_hands_candidate.py';path.write_text(candidate);ast.parse(candidate)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=load('old',ROOT/'FahadAR/occlusion.py');new=load('new',path)
checks={}
def setup():
    p=np.zeros((33,4));p[:,:2]=.5;p[:,3]=0
    for i,xy in {11:(.60,.28),12:(.40,.28),23:(.59,.70),24:(.41,.70),13:(.63,.51),15:(.49,.54)}.items():p[i,:2]=xy;p[i,3]=.95
    w=np.zeros((33,4));w[:,3]=1;w[[13,15],2]=.25
    skin=np.zeros((360,640),np.float32)
    a,b=[tuple(np.round(p[i,:2]*[640,360]).astype(int)) for i in (13,15)]
    cv2.line(skin,a,b,.95,14);cv2.circle(skin,b,9,.95,-1)
    return p,w,skin
def run(m,p,w,skin,long=False):return m.compose_matte(p,w,[],{},dict(skin=skin,head=np.zeros_like(skin)),long)[0]
def at(mask,p,i=15):x,y=np.round(p[i,:2]*[640,360]).astype(int);return int(mask[y,x])
p,w,skin=setup()
checks['old_code_reproduces_hidden_visible_hand']=at(run(old,p,w,skin),p)==0
checks['visible_hand_overrides_wrong_depth']=at(run(new,p,w,skin),p)>230
p[15,3]=.3
checks['low_visibility_wrist_keeps_connected_skin']=at(run(new,p,w,skin),p)>230
p,w,skin=setup();skin[:]=0
checks['no_actual_skin_no_recovery']=not run(new,p,w,skin).any()
skin[:]=1
checks['skin_coloured_field_not_restored']=np.array_equal(run(old,p,w,skin),run(new,p,w,skin))
p,w,skin=setup();long=run(new,p,w,skin,True)
mid=(p[13,:2]+p[15,:2])*.5;x,y=np.round(mid*[640,360]).astype(int)
checks['full_sleeve_forearm_covered']=long[y,x]==0
# Full sleeves require a distinct hand-sized component rather than an entire
# bare-forearm component; detailed hand evidence is verified by old suite.
p,w,skin=setup();skin[:]=0;x,y=np.round(p[15,:2]*[640,360]).astype(int);cv2.circle(skin,(x,y),7,.95,-1)
checks['full_sleeve_visible_hand_only']=at(run(new,p,w,skin,True),p)>230
p,w,skin=setup();p[[13,15],0]+=.25;skin[:]=0
a,b=[tuple(np.round(p[i,:2]*[640,360]).astype(int)) for i in (13,15)];cv2.line(skin,a,b,.95,14)
checks['outside_shirt_matte_unchanged']=np.array_equal(run(old,p,w,skin),run(new,p,w,skin))
p,w,skin=setup();p[15,3]=0
checks['missing_hand_not_invented']=np.array_equal(run(old,p,w,skin),run(new,p,w,skin))
def marks(a):return [NS(x=x,y=y,z=z,visibility=v) for x,y,z,v in a]
p,w,skin=setup();rgb=np.full((360,640,3),[112,158,146],np.uint8);rgb[skin>.5]=[132,128,116]
r=NS(pose_landmarks=marks(p),pose_world_landmarks=marks(w),face_landmarks=[],left_hand_landmarks=[],right_hand_landmarks=[])
for label,m in [('old',old),('new',new)]:
    f=m.FastForeground();f.reference=(np.array([137.,118.]),np.array([5.,5.]));f.last_face=time.monotonic()
    mask,_=f.process(rgb,r);checks[label+'_screenshot_colours_hand_visible']=at(mask,p)>200
checks['reproduces_screenshot_colours_failure']=not checks.pop('old_screenshot_colours_hand_visible')
# Run the prior 19 checks against this candidate, preserving full sleeves,
# near/far profiles and colour-reference behavior.
test=(TMP/'ar_studio_inspect_20260915.py').read_text().replace("TMP/'ar_studio_occlusion_candidate.py'","TMP/'ar_hands_candidate.py'")
exec(compile(test,str(TMP/'ar_studio_inspect_20260915.py'),'exec'),{'__name__':'__main__','__file__':str(TMP/'ar_studio_inspect_20260915.py')})
off=cv2.imread(str(TMP/'ar_hand_layer_off.png'));on=cv2.imread(str(TMP/'ar_hand_layer_on.png'))
if off is not None and on is not None:
    patch=on[110:240,250:390].astype(float)
    checks['blender_layer_changes_shirt_pixels']=bool(np.count_nonzero(np.any(off[110:240,250:390]!=on[110:240,250:390],axis=2))>100)
    checks['blender_foreground_patch_is_flat_camera_colour']=bool(np.max(patch.std(axis=(0,1)))<2)
checks={k:bool(v) for k,v in checks.items()}
print('HANDS_REPORT',json.dumps(checks,indent=2))
(TMP/'ar_hands_report.json').write_text(json.dumps(checks,indent=2))
for label in ('off','on'):
    im=TMP/('ar_hand_layer_'+label+'.png')
    if im.exists():print('IMAGE '+base64.b64encode(cv2.imencode('.png',cv2.resize(cv2.imread(str(im)),(480,270)))[1]).decode())
assert all(v is True for v in checks.values())
