import sys,os,json,time
sys.dont_write_bytecode=True
from pathlib import Path
import cv2,numpy as np,mediapipe as mp
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'FahadAR'));sys.path.insert(0,str(ROOT/'FahadLive'))
from female_coverage import measure
from fahad_backend import points
model=Path(os.environ['LOCALAPPDATA'])/'PoseCap/backends/mediapipe/models/holistic_landmarker.task'
options=mp.tasks.vision.HolisticLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),output_segmentation_mask=True)
rows=[];tiles=[]
with mp.tasks.vision.HolisticLandmarker.create_from_options(options) as detector:
 for second in [14,20,41,54,62]:
  frame=cv2.imread(str(Path(__file__).with_name(f'frame_{second:02}.jpg')))
  crop=frame[168:824,207:1373].copy();crop[:,960:]=0
  rgb=cv2.cvtColor(cv2.resize(crop,(640,360)),cv2.COLOR_BGR2RGB)
  start=time.perf_counter();result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb));ms=(time.perf_counter()-start)*1000
  pose=points(result.pose_landmarks);mask=result.segmentation_mask.numpy_view() if result.segmentation_mask else None
  evidence=measure(mask,pose,float(second));view=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
  for y,lo,hi in evidence['rows']:cv2.line(view,(round(lo),round(y)),(round(hi),round(y)),(0,240,0),1)
  if len(pose)==33:
   for s,e,w in [(11,13,15),(12,14,16)]:
    for a,b in [(s,e),(e,w)]:cv2.line(view,tuple(np.rint(np.array(pose[a][:2])*[640,360]).astype(int)),tuple(np.rint(np.array(pose[b][:2])*[640,360]).astype(int)),(255,190,0),1)
  cv2.putText(view,f'{second}s - composited reference only',(8,24),0,.55,(255,255,255),1)
  tiles.append(view);rows.append({'second':second,'model_ms':ms,'evidence':evidence,'wrists':[pose[j] for j in (15,16)] if len(pose)==33 else []})
Path(__file__).with_name('video_model_inspection.json').write_text(json.dumps(rows,indent=2))
cv2.imwrite(str(Path(__file__).with_name('video_model_inspection.jpg')),np.vstack(tiles))
print(json.dumps(rows))
