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
from occlusion import Foreground,FastForeground,MaskStabilizer
from latest_camera import LatestCamera
from stability import LandmarkLock,QuietRegionLock
from female_stability import SleeveLock
from female_maskflow import MaskStabilizer as FemaleMaskStabilizer
from female_regional_flow import ImageMotion
from female_reliability import RegionalLock
from female_arm_observations import ArmTranslationRepair
from female_arm_image import ArmImageFollower
from female_arm_stabilizer import ArmStabilizer
from female_continuity import motion_confirmed

def send(conn,meta,pixels=b''):
    data=json.dumps(meta,separators=(',',':')).encode()
    conn.sendall(struct.pack('!II',len(data),len(pixels))+data+pixels)

def female_foreground(foreground,stabilizer,rgb,result,segmentation):
    """Same frame/landmarks; overlap independent matte work with pose solving.

    The caller always joins before publishing, changing profile, recalibrating,
    or processing another frame. This never queues future camera frames.
    """
    started=time.perf_counter()
    mask,info=foreground.compose(result,segmentation.result()) if segmentation else foreground.process(rgb,result)
    mask=stabilizer.apply(rgb,mask)
    return mask,info,(time.perf_counter()-started)*1000

def main():
    a=argparse.ArgumentParser();a.add_argument('--port',type=int,required=True)
    a.add_argument('--model',required=True);a.add_argument('--camera',type=int,default=0)
    a.add_argument('--quality',choices=['smooth','semantic'],default='smooth')
    a.add_argument('--sleeves',choices=['short','long'],default='short')
    a.add_argument('--stability',choices=['best','standard'],default='best')
    a.add_argument('--profile',choices=['female','western','male_eastern','other'],default='other')
    args=a.parse_args();conn=socket.create_connection(('127.0.0.1',args.port),timeout=15);conn.settimeout(3)
    cap=None;detector=None;foreground=None;pool=None;capture=None;female_pool=None
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
        solver=Solver();solver.best_stability=args.stability=='best';solver.full_sleeves=args.sleeves=='long'
        solver.western_stability=args.profile in ('western','male_eastern')
        solver.male_livefit=args.profile in ('western','male_eastern')
        solver.female_livefit=args.profile=='female'
        solver.male_continuity_enabled=solver.male_livefit
        foreground.male_neck=solver.male_livefit
        detailed=args.profile in ('female','western','male_eastern')
        female_profile=args.profile=='female';female_motion=ImageMotion()
        female_regions=RegionalLock();female_mask=FemaleMaskStabilizer()
        female_arms=ArmTranslationRepair()
        female_image_arms=ArmImageFollower();female_sleeves=ArmStabilizer()
        from male_continuity import ImageMotion as MaleImageMotion
        male_motion=MaleImageMotion()
        body_lock=LandmarkLock(preserve_confidence=detailed);quiet_lock=QuietRegionLock();mask_lock=MaskStabilizer();sleeve_lock=SleeveLock()
        start=time.monotonic();stamp=0;seq=0;previous_pose=[];commands=b''
        while True:
            if select.select([conn],[],[],0)[0]:
                command=conn.recv(128)
                if not command:break
                commands+=command;lines=commands.split(b'\n');commands=lines.pop()
                if b'STOP' in lines:break
                for line in lines:
                    if line==b'CAL':
                        solver.reset_calibration()
                        body_lock.reset()
                        quiet_lock.reset()
                        sleeve_lock.reset()
                        mask_lock.reset()
                        female_motion.reset()
                        female_regions.reset();female_mask.reset();female_arms.reset();female_image_arms.reset();female_sleeves.reset()
                        male_motion.reset()
                        if hasattr(foreground,'reset'):foreground.reset()
                    elif line in (b'SLEEVE_LONG',b'SLEEVE_SHORT'):
                        foreground.full_sleeves=line==b'SLEEVE_LONG'
                        solver.full_sleeves=foreground.full_sleeves
                    elif line in (b'STABILITY_BEST',b'STABILITY_STANDARD'):
                        solver.best_stability=line==b'STABILITY_BEST';body_lock.reset();quiet_lock.reset()
                        female_motion.reset()
                        female_regions.reset();female_mask.reset();female_arms.reset();female_image_arms.reset();female_sleeves.reset()
                        male_motion.reset();solver.male_heading=None;solver.male_motion_active=False
                    elif line in (b'PROFILE_FEMALE',b'PROFILE_WESTERN',b'PROFILE_MALE_EASTERN',b'PROFILE_OTHER'):
                        solver.western_stability=line in (b'PROFILE_WESTERN',b'PROFILE_MALE_EASTERN')
                        solver.male_livefit=solver.western_stability;foreground.male_neck=solver.male_livefit
                        female_profile=line==b'PROFILE_FEMALE';female_motion.reset()
                        female_regions.reset();female_mask.reset();female_arms.reset();female_image_arms.reset();female_sleeves.reset()
                        solver.female_livefit=female_profile;solver.female_heading=None;solver.female_motion_active=False
                        solver.male_continuity_enabled=solver.male_livefit
                        male_motion.reset();solver.male_heading=None;solver.male_motion_active=False
                        detailed=line in (b'PROFILE_FEMALE',b'PROFILE_WESTERN',b'PROFILE_MALE_EASTERN');body_lock.preserve_confidence=detailed
                        body_lock.reset();quiet_lock.reset();sleeve_lock.reset()
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
            model_done=time.perf_counter()
            matte_job=None
            if female_profile and solver.best_stability:
                if female_pool is None:female_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='same_frame_female_matte')
                matte_job=female_pool.submit(female_foreground,foreground,female_mask,rgb,result,segmentation)
            raw={'t':elapsed,'pose':points(result.pose_landmarks),'world':points(result.pose_world_landmarks),
                 'face':points(result.face_landmarks),'left_hand':points(result.left_hand_world_landmarks),
                 'right_hand':points(result.right_hand_world_landmarks)}
            observed_pose=raw['pose'];observed_world=raw['world']
            motion=female_motion.update(rgb,observed_pose,captured) if female_profile and solver.best_stability else None
            male_evidence=male_motion.update(rgb,observed_pose,captured) if solver.male_continuity_enabled and solver.best_stability else None
            if motion is not None:
                raw['pose'],raw['world']=female_image_arms.update(raw['pose'],raw['world'],captured,motion)
                motion=female_image_arms.motion_evidence(motion)
                raw['pose'],raw['world']=female_arms.update(raw['pose'],raw['world'],captured,motion)
            else:female_arms.reset();female_image_arms.reset()
            locked=body_lock.update(raw['pose'],raw['world'],captured) if solver.best_stability else None
            tracking_valid=locked is not None if solver.best_stability else len(raw['pose'])==33
            if locked:raw['pose'],raw['world']=quiet_lock.update(*locked)
            if locked and detailed:raw['pose'],raw['world']=sleeve_lock.update(raw['pose'],raw['world'])
            if locked and female_profile and solver.best_stability:
                raw['pose'],raw['world']=female_image_arms.retain_motion(raw['pose'],raw['world'],body_lock,sleeve_lock)
                raw['pose'],raw['world']=female_regions.update(raw['pose'],raw['world'],observed_pose,captured,motion)
                female_image_arms.commit(raw['pose'],raw['world'],captured)
                # After commit: the image follower keeps its own unfiltered
                # reference; only the solver input is sleeve-stabilized.
                raw['pose'],raw['world']=female_sleeves.update(raw['pose'],raw['world'],captured,motion)
            elif female_profile:
                female_regions.reset();female_arms.reset();female_image_arms.reset();female_sleeves.reset()
            solver.female_motion_active=motion_confirmed(motion,captured)
            solver.male_motion_active=motion_confirmed(male_evidence,captured)
            payload,info=solver.solve(raw,16/9)
            pose_done=time.perf_counter()
            if matte_job is not None:
                mask,mask_info,matte_work_ms=matte_job.result()
            else:
                mask,mask_info=foreground.compose(result,segmentation.result()) if segmentation else foreground.process(rgb,result)
                if solver.best_stability:mask=(female_mask if female_profile else mask_lock).apply(rgb,mask)
            matte_done=time.perf_counter()
            previous_pose=raw['pose']
            send(conn,{'kind':'frame','seq':seq,'captured_at':captured,'pose':asdict(payload),
                       'landmarks':raw['pose'],'world':raw['world'],'info':info,'tracking_valid':tracking_valid,
                       **({'observed_landmarks':observed_pose,'observed_world':observed_world} if solver.male_livefit else {}),
                       **({'female_motion':motion,'female_regions_held':female_regions.held,'female_region_state':female_regions.state,'female_mask_flow_area':female_mask.flow_area,
                           'female_arm_observations':female_arms.state,
                           'female_arm_image':female_image_arms.state,
                           'observed_landmarks':observed_pose,'observed_world':observed_world} if motion is not None else {}),
                       **({'female_stages_ms':{'crop_and_model':(model_done-inference_start)*1000,
                           'motion_and_pose':(pose_done-model_done)*1000,'foreground':(matte_done-pose_done)*1000,
                           'foreground_work':matte_work_ms,'same_frame_parallel':True}} if motion is not None else {}),
                       **({'male_motion':male_evidence,'male_stages_ms':{'crop_and_model':(model_done-inference_start)*1000,
                           'motion_and_pose':(pose_done-model_done)*1000,'foreground':(matte_done-pose_done)*1000}} if male_evidence is not None else {}),
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
        if female_pool is not None:female_pool.shutdown(wait=True,cancel_futures=True)
        if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
        if detector is not None:detector.close()
        if foreground is not None:foreground.close()
        conn.close()

if __name__=='__main__':main()
