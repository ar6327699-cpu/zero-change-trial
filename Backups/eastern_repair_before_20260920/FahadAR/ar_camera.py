"""Local-only synchronized webcam + pose sender for the separate AR experiment.

No video files, uploads or changes to the original PoseCap backend. RGB and its
own pose travel in a single packet so garment/video timestamps cannot drift.
"""
import argparse,json,socket,struct,time,select,sys
from dataclasses import asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'FahadLive'))
from fahad_backend import Solver,points
from occlusion import Foreground,FastForeground
from latest_camera import LatestCamera

def send(conn,meta,pixels=b''):
    data=json.dumps(meta,separators=(',',':')).encode()
    conn.sendall(struct.pack('!II',len(data),len(pixels))+data+pixels)

def main():
    a=argparse.ArgumentParser();a.add_argument('--port',type=int,required=True)
    a.add_argument('--model',required=True);a.add_argument('--camera',type=int,default=0)
    a.add_argument('--quality',choices=['smooth','semantic'],default='smooth')
    a.add_argument('--sleeves',choices=['short','long'],default='short')
    args=a.parse_args();conn=socket.create_connection(('127.0.0.1',args.port),timeout=15);conn.settimeout(3)
    cap=None;detector=None;foreground=None;pool=None;capture=None
    try:
        send(conn,{'kind':'status','message':'Loading existing local pose model...'})
        import mediapipe as mp
        opt=mp.tasks.vision.HolisticLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=args.model),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=.5,min_pose_landmarks_confidence=.5,min_hand_landmarks_confidence=.4)
        detector=mp.tasks.vision.HolisticLandmarker.create_from_options(opt)
        foreground=Foreground(full_sleeves=args.sleeves=='long') if args.quality=='semantic' else FastForeground(full_sleeves=args.sleeves=='long')
        if args.quality=='semantic':pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='same_frame_skin')
        cap=cv2.VideoCapture(args.camera,cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,640);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        cap.set(cv2.CAP_PROP_FPS,30);cv2.setNumThreads(2)
        if not cap.isOpened():raise RuntimeError('Camera unavailable. Close other camera apps, then Start AR again.')
        capture=LatestCamera(cap)
        solver=Solver();start=time.monotonic();stamp=0;seq=0;previous_pose=[]
        while True:
            if select.select([conn],[],[],0)[0]:
                command=conn.recv(128)
                if not command or b'STOP' in command:break
                if b'CAL' in command:
                    solver.reset_calibration()
                    if hasattr(foreground,'reset'):foreground.reset()
            bgr,captured=capture.read();inference_start=time.perf_counter()
            # Fixed 16:9 cover crop: coordinates and displayed video use the
            # exact same crop, including cameras that ignore requested sizes.
            h,w=bgr.shape[:2];wanted=w*9/16
            if h>wanted:
                y=int((h-wanted)/2);bgr=bgr[y:y+int(wanted)]
            else:
                wanted=h*16/9;x=int((w-wanted)/2);bgr=bgr[:,x:x+int(wanted)]
            rgb=cv2.cvtColor(cv2.resize(bgr,(640,360)),cv2.COLOR_BGR2RGB)
            elapsed=time.monotonic()-start;stamp=max(stamp+1,int(elapsed*1000))
            segmentation=pool.submit(foreground.infer,rgb,previous_pose,stamp) if pool else None
            result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),stamp)
            raw={'t':elapsed,'pose':points(result.pose_landmarks),'world':points(result.pose_world_landmarks),
                 'face':points(result.face_landmarks),'left_hand':points(result.left_hand_world_landmarks),
                 'right_hand':points(result.right_hand_world_landmarks)}
            payload,info=solver.solve(raw,16/9)
            mask,mask_info=foreground.compose(result,segmentation.result()) if segmentation else foreground.process(rgb,result)
            previous_pose=raw['pose']
            send(conn,{'kind':'frame','seq':seq,'captured_at':captured,'pose':asdict(payload),
                       'landmarks':raw['pose'],'world':raw['world'],'info':info,
                       'occlusion':mask_info,'pixel_format':'rgb+matte8',
                       'inference_ms':(time.perf_counter()-inference_start)*1000},rgb.tobytes()+mask.tobytes())
            seq+=1
    except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
    except Exception as e:
        try:send(conn,{'kind':'error','message':str(e)})
        except OSError:pass
        raise
    finally:
        if capture is not None:capture.close()
        elif cap is not None:cap.release()
        if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
        if detector is not None:detector.close()
        if foreground is not None:foreground.close()
        conn.close()

if __name__=='__main__':main()
