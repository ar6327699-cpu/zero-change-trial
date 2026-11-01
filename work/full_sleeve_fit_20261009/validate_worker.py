"""Exercise the real local sender using supplied recorded frames, no webcam."""
import sys,json,time,socket,struct,threading,os
sys.dont_write_bytecode=True
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'FahadAR'))
import ar_camera
frames=[]
for second in [14,20,41,54,62]:
 frame=cv2.imread(str(Path(__file__).with_name(f'frame_{second:02}.jpg')))
 crop=frame[168:824,207:1373].copy();crop[:,960:]=0;frames.append(cv2.resize(crop,(640,360)))
class RecordedCamera:
 def __init__(self,*args,**kw):self.started=time.monotonic();self.open=True
 def set(self,*args):return True
 def isOpened(self):return self.open
 def read(self):
  time.sleep(.033)
  return (True,frames[int((time.monotonic()-self.started)/.55)%len(frames)].copy()) if self.open else (False,None)
 def release(self):self.open=False
ar_camera.cv2.VideoCapture=RecordedCamera
server=socket.socket();server.bind(('127.0.0.1',0));server.listen(1);server.settimeout(30)
received=[];errors=[]
def receiver():
 try:
  conn,_=server.accept();conn.settimeout(30)
  def read(n):
   data=b''
   while len(data)<n:
    part=conn.recv(n-len(data))
    if not part:raise EOFError()
    data+=part
   return data
  while True:
   a,b=struct.unpack('!II',read(8));meta=json.loads(read(a));pixels=read(b)
   if meta.get('kind')=='error':raise RuntimeError(meta)
   if meta.get('kind')!='frame':continue
   assert len(pixels)==640*360*4 and len(meta['landmarks'])==33
   assert meta['female_coverage']['stamp']==meta['captured_at']
   assert np.isfinite(meta['landmarks']).all()
   received.append(meta)
   if len(received)==15:conn.sendall(b'STOP\n')
 except EOFError:pass
 except Exception as e:errors.append(repr(e))
 finally:server.close()
t=threading.Thread(target=receiver);t.start()
model=Path(os.environ['LOCALAPPDATA'])/'PoseCap/backends/mediapipe/models/holistic_landmarker.task'
sys.argv=['ar_camera.py','--port',str(server.getsockname()[1]),'--model',str(model),'--camera','0','--quality','smooth','--sleeves','long','--stability','best','--profile','female']
ar_camera.main();t.join(timeout=35)
assert not errors and len(received)>=15,(errors,len(received))
assert any(len(m['female_coverage']['rows'])>=3 for m in received)
Path(__file__).with_name('worker_packets.json').write_text(json.dumps(received))
report={'packets':len(received),'matched_rgb_pose_mask_timestamps':True,'visible_outline_available':True,'webcam_opened':False,'source':'User-supplied composited recording, used only for integration smoke test','median_worker_ms':float(np.median([m['inference_ms'] for m in received]))}
Path(__file__).with_name('worker_validation.json').write_text(json.dumps(report,indent=2));print('WORKER_VALIDATION',json.dumps(report))
