"""Local camera solver for Fahad's 52-bone Mixamo binding. No uploads.

The showroom stays in place but may turn around the vertical axis. Parent-local
limb rotations compensate that heading consistently. No pelvis roll/pitch is
allowed to leak into both legs. Head and hands use their existing calibration.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import logging
import math
from pathlib import Path
import time
import numpy as np
import cv2
from posecap_contracts import PoseFrame, PosePayload, SCHEMA_VERSION
from posecap_core.skeleton import BODY_JOINT_NAMES

I = np.eye(3)
ZERO = [0., 0., 0.]
SOLVER_VERSION = 'side-profile-1'

def blend_rotation(a, b, weight):
    return a @ cv2.Rodrigues(rv(a.T @ b)*float(np.clip(weight, 0, 1)))[0]

def ramp(value, low, high):
    v = float(np.clip((value-low)/(high-low), 0, 1))
    return v*v*(3-2*v)

def yaw_angle(R):
    return math.atan2(-R[2, 0], R[0, 0])

def yaw_rotation(angle):
    return cv2.Rodrigues(np.array([0., angle, 0.]))[0]

def unit(v):
    v = np.asarray(v, dtype=float)
    return v / max(float(np.linalg.norm(v)), 1e-8)

def rotation(a, b):
    a, b = unit(a), unit(b)
    if np.linalg.norm(b) < .5:
        return I.copy()
    c = float(np.clip(a @ b, -1, 1)); cross = np.cross(a, b)
    if c < -.9999:
        axis = unit(np.cross(a, [0, 0, 1]))
        return cv2.Rodrigues(axis * math.pi)[0]
    K = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    return I + K + K @ K / max(1 + c, 1e-8)

def rv(R):
    return cv2.Rodrigues(np.asarray(R, dtype=float))[0].reshape(3)

def frame(x, y):
    x = unit(x); y = unit(y - x * np.dot(x, y)); z = unit(np.cross(x, y))
    return np.column_stack([x, np.cross(z, x), z])

def mean_rotation(matrices):
    u, _, vt = np.linalg.svd(np.mean(matrices, axis=0))
    return u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt

def angle(a, b):
    return math.acos(float(np.clip(unit(a) @ unit(b), -1, 1)))

def solve_forearm_roll(points, side, forearm):
    """Measured palm orientation split between forearm roll and wrist bend.

    Rotating around the rest forearm X axis cannot change the elbow-to-wrist
    direction. Head, shoulder and leg solving remain untouched.
    """
    if len(points) != 21:
        return None
    p = np.asarray(points, dtype=float)[:, :3].copy()
    p[:, 1:] *= -1  # MediaPipe world coordinates -> source camera coordinates.
    long_axis, across = p[9]-p[0], p[17]-p[5]
    if (not np.isfinite(p).all() or np.linalg.norm(long_axis)<.01
            or np.linalg.norm(across)<.01
            or np.linalg.norm(np.cross(unit(long_axis),unit(across)))<.25):
        return None
    sx = 1 if side == 'left' else -1
    # Mixamo palm: distal +/-X, pinky direction world +Y (source -Z).
    rest_palm = np.array([[sx, 0, 0], [0, 0, sx], [0, -1, 0.]])
    hand_global = frame(long_axis, across) @ rest_palm.T
    relative = forearm.T @ hand_global
    vector = rv(relative); theta = np.linalg.norm(vector)
    qw = math.cos(theta*.5)
    qx = float(vector[0] * math.sin(theta*.5)/theta) if theta>1e-8 else 0.
    if math.hypot(qw,qx)<1e-6:
        return None  # Swing-twist is indeterminate at an exact 180-degree swing.
    twist_angle = (2*math.atan2(qx,qw)+math.pi)%(2*math.pi)-math.pi
    twist_angle = float(np.clip(twist_angle, -2.88, 2.88))
    twist = cv2.Rodrigues(np.array([twist_angle,0.,0.]))[0]
    wrist = rv(twist.T @ relative)
    wrist_limit = math.radians(75)
    if np.linalg.norm(wrist)>wrist_limit:
        wrist *= wrist_limit/np.linalg.norm(wrist)
    return twist, cv2.Rodrigues(wrist)[0]

def solve_hand(points, side):
    """Source axes explicitly verified against this file's rest-compensated rig."""
    out = np.zeros((15, 3))
    if len(points) != 21:
        return out
    p = np.asarray(points)[:, :3]
    sign = -1 if side == 'left' else 1
    bends = []
    for base in (5, 9, 17, 13):
        palm = p[base] - p[0]
        a, b, c = p[base+1]-p[base], p[base+2]-p[base+1], p[base+3]-p[base+2]
        bends.append([angle(palm, a), angle(a, b), angle(b, c)])
    bends = np.asarray(bends)
    # MediaPipe often underestimates occluded MCPs in a real fist. Measured PIP
    # and DIP bends supply a conservative lower bound, not a body-gesture guess.
    for i, (mcp, pip, dip) in enumerate(bends):
        closure = float(np.clip((pip + dip - .35) / 2.0, 0, 1))
        mcp = max(mcp - .08, closure * 1.10)
        # The actual mesh palm lies in world XY, NOT world XZ. Therefore
        # flexion is source Z (world -Y), not source Y (sideways curling).
        out[3*i:3*i+3, 2] = sign * np.clip([mcp, pip, dip], 0, [1.35, 1.6, 1.4])
    # Thumb opposition has two axes. Retarget its measured segment directions
    # in the hand's own palm frame instead of guessing it from finger closure.
    palm = frame(p[9]-p[0], p[17]-p[5])
    sx = -sign
    target_palm = np.array([[sx, 0, 0], [0, 0, sx], [0, -1, 0.]])
    rest_thumb = np.array([sx*.774596669, -.447213595, .447213595])
    parent = I.copy()
    for i in range(3):
        desired = target_palm @ (palm.T @ (p[i+2]-p[i+1]))
        global_rotation = rotation(rest_thumb, desired)
        v = rv(parent.T @ global_rotation)
        limit = 1.4 if i == 0 else 1.5
        if np.linalg.norm(v)>limit: v *= limit/np.linalg.norm(v)
        out[12+i] = v
        parent = parent @ cv2.Rodrigues(v)[0]
    return out

class Solver:
    def __init__(self):
        self.reset_calibration()
        self.filtered = {}
        self.last_good = {}
        self.last_t = None

    def reset_calibration(self):
        self.head_samples = []; self.chest_samples = []
        self.head_zero = None; self.chest_zero = None
        self.spatial_samples = []; self.spatial_zero = None
        self.heading = 0.; self.heading_seen = -1e6
        self.female_heading = None
        self.male_heading = None

    def male_continuity_active(self):
        return (getattr(self,'male_continuity_enabled',False) and getattr(self,'male_livefit',False)
                and getattr(self,'best_stability',False) and not getattr(self,'female_livefit',False))

    def smooth(self, key, value, dt):
        if key=='pelvis' and self.male_continuity_active():
            # The heading follower already filters the male root. Local
            # sleeve rotations must use exactly that same root this packet.
            self.filtered[key]=value.copy()
            return value.copy()
        old = self.filtered.get(key, I)
        delta = rv(old.T @ value)
        # Approximately 65ms time constant; one filter, not stacked addon filters.
        angle=float(np.linalg.norm(delta))
        if getattr(self,'best_stability',False):
            sleeve_joint=getattr(self,'full_sleeves',False) and key in (
                'left_shoulder','right_shoulder','left_elbow','right_elbow')
            # Anchor-relative hysteresis: quiet arm noise stays still, but a
            # slow intentional move eventually crosses the same fixed anchor.
            deadband=math.radians(2.0 if sleeve_joint else .35)
            female_motion=getattr(self,'female_livefit',False) and getattr(self,'female_motion_active',False) and sleeve_joint
            if female_motion:deadband=math.radians(.6)
            male_motion=self.male_continuity_active() and getattr(self,'male_motion_active',False) and key in (
                'left_shoulder','right_shoulder','left_elbow','right_elbow')
            if male_motion:deadband=math.radians(.6 if sleeve_joint else .35)
            if angle<deadband:return old.copy()
            # Speed-aware bone damping; do not stack a slow filter on fast arms.
            tau=.09 if angle<math.radians(3) else .035
            if female_motion:tau=.035 if angle<math.radians(3) else .018
            if male_motion:tau=.035 if angle<math.radians(3) else .018
            if angle>math.radians(110) and dt<.18:
                delta*=math.radians(110)/angle
            alpha=1-math.exp(-dt/tau)
        else:alpha = 1 - math.exp(-dt / .065)
        out = old @ cv2.Rodrigues(delta * alpha)[0]
        self.filtered[key] = out
        return out

    def solve(self, raw, aspect=16/9):
        t = float(raw['t']); dt = min(.25, max(.01, t - self.last_t)) if self.last_t is not None else .1
        self.last_t = t
        p = np.asarray(raw.get('pose', []), dtype=float)
        w = np.asarray(raw.get('world', []), dtype=float)
        rots = {n: I.copy() for n in BODY_JOINT_NAMES}
        info = {'left_leg': False, 'right_leg': False, 'left_hand': False, 'right_hand': False,
                'head_calibrated': self.head_zero is not None, 'body': len(p) == 33}
        def visible(indices, threshold=.5):
            return len(p) == 33 and all(p[j, 3] >= threshold and .01 < p[j, 0] < .99 and .01 < p[j, 1] < .99 for j in indices)
        q = None; spatial = None; profile = 0.
        if len(p) == len(w) == 33:
            q = p[:, :3].copy(); q[:, 0] = (q[:, 0]-.5) * aspect; q[:, 1] = .5-q[:, 1]
            world = w[:, :3].copy(); world[:, 1:] *= -1
            # World shoulder/hip width retains depth at 90 degrees. Screen
            # shoulder width does not; it must not scale the profile depth.
            # In a genuine profile the far shoulder/hip is often marked
            # occluded. Require the near torso, not all four front landmarks.
            torso_seen = (visible([11],.5) or visible([12],.5)) and (visible([23],.5) or visible([24],.5))
            female_live=getattr(self,'female_livefit',False) and getattr(self,'best_stability',False)
            if female_live:torso_seen=visible([11,23],.5) or visible([12,24],.5)
            male_continuity=self.male_continuity_active()
            if male_continuity:torso_seen=visible([11,23],.5) or visible([12,24],.5)
            if torso_seen and np.isfinite(world).all():
                up = (world[11]+world[12]-world[23]-world[24])*.5
                if min(np.linalg.norm(world[11]-world[12]), np.linalg.norm(up))>.08:
                    spatial = frame(world[11]-world[12], up)
                    axis = .65*unit(world[23]-world[24])+.35*unit(world[11]-world[12])
                    heading_frame = frame(axis, np.array([0.,1.,0.]))
                    if self.spatial_zero is None and visible([11,12,23,24], .5):
                        self.spatial_samples.append((spatial, heading_frame))
                        if len(self.spatial_samples)>=20:
                            recent = self.spatial_samples[-20:]
                            average = mean_rotation([x[0] for x in recent])
                            if max(np.linalg.norm(rv(average.T@x[0])) for x in recent)<math.radians(12):
                                self.spatial_zero = (average, mean_rotation([x[1] for x in recent]))
                    if self.spatial_zero is not None:
                        target = yaw_angle(heading_frame@self.spatial_zero[1].T)
                        # A quiet front dead zone preserves independent leg lifts.
                        if male_continuity:
                            from male_continuity import HeadingFollower
                            if self.male_heading is None:self.male_heading=HeadingFollower()
                            self.heading=self.male_heading.update(self.heading,target,dt)
                        elif getattr(self,'western_stability',False):
                            from western_tracking import follow_heading
                            self.heading=follow_heading(self.heading,target,dt)
                        elif female_live:
                            from female_continuity import HeadingFollower
                            if self.female_heading is None:self.female_heading=HeadingFollower()
                            self.heading=self.female_heading.update(self.heading,target,dt)
                        else:
                            target *= ramp(abs(target), math.radians(15), math.radians(45))
                            difference = (target-self.heading+math.pi)%(2*math.pi)-math.pi
                            difference = np.clip(difference, -dt*math.radians(180), dt*math.radians(180))
                            self.heading += float(difference)*(1-math.exp(-dt/.10))
                        self.heading_seen = t
            # Brief loss keeps the last heading, rather than snapping front.
            if not torso_seen and t-self.heading_seen>1.0 and not getattr(self,'best_stability',False):
                self.heading *= math.exp(-dt/.5)
            profile = ramp(abs(self.heading), math.radians(12), math.radians(55))
            if female_live:
                # Returning front after a full revolution is front again.
                wrapped=(self.heading+math.pi)%(2*math.pi)-math.pi
                profile=ramp(abs(wrapped),math.radians(12),math.radians(55))
            if male_continuity:
                wrapped=(self.heading+math.pi)%(2*math.pi)-math.pi
                profile=ramp(abs(wrapped),math.radians(12),math.radians(55))
            screen_width = np.linalg.norm(q[11, :2]-q[12, :2])
            world_width = np.linalg.norm(w[11, :3]-w[12, :3])
            scale = np.clip(screen_width/max(world_width, .05), .15, 3)
            q[:, 2] = -w[:, 2] * scale * .65
            image_torso = (q[11,:2]+q[12,:2]-q[23,:2]-q[24,:2])*.5
            world_torso = (world[11]+world[12]-world[23]-world[24])*.5
            side_scale = np.clip(np.linalg.norm(image_torso)/max(np.linalg.norm(world_torso),.08),.15,3)
            if getattr(self,'male_livefit',False):
                pair_scales=[np.linalg.norm(q[a,:2]-q[b,:2])/max(np.linalg.norm(world[a]-world[b]),.08)
                             for a,b in ((11,23),(12,24)) if visible([a,b],.5)]
                if pair_scales:side_scale=float(np.clip(np.median(pair_scales),.15,3))
            origin = (q[23]+q[24])*.5
            side_q = (world-(world[23]+world[24])*.5)*side_scale+origin
            q = (1-profile)*q+profile*side_q
            if getattr(self,'male_livefit',False):
                # Monocular world x/y is inferred and disagrees with visible
                # elbows at profile angles. Keep CURRENT observed screen x/y;
                # use world depth only. Far unobserved joints retain the model.
                seen=(p[:,3]>=.5)&(p[:,0]>.01)&(p[:,0]<.99)&(p[:,1]>.01)&(p[:,1]<.99)
                q[seen,0]=(p[seen,0]-.5)*aspect;q[seen,1]=.5-p[seen,1]
            elif female_live:
                # The visible shoulder/elbow/wrist must meet THIS camera frame
                # at a side view; inferred world X/Y otherwise pulls it away.
                seen=(p[:,3]>=.5)&(p[:,0]>.01)&(p[:,0]<.99)&(p[:,1]>.01)&(p[:,1]<.99)
                q[seen,0]=(p[seen,0]-.5)*aspect;q[seen,1]=.5-p[seen,1]
        root = yaw_rotation(self.heading)
        chest = I.copy()
        if q is not None and visible([11, 12, 23, 24], .5):
            measured = frame(q[11]-q[12], (q[11]+q[12]-q[23]-q[24])*.5)
            if self.chest_zero is None:
                self.chest_samples.append(measured)
                if len(self.chest_samples) >= 20:
                    self.chest_zero = mean_rotation(self.chest_samples[-20:])
            if self.chest_zero is not None:
                v = np.clip(rv(measured @ self.chest_zero.T), [-.7, -.9, -.6], [.7, .9, .6])
                chest = cv2.Rodrigues(v)[0]
        if spatial is not None and self.spatial_zero is not None:
            spatial_chest = spatial@self.spatial_zero[0].T
            # Separate global heading from the small torso bend/twist. A pure
            # side turn should rotate the hips, torso, feet and head together.
            residual = rv(root.T@spatial_chest)
            residual = np.clip(residual, [-.7,-.65,-.6], [.7,.65,.6])
            chest = blend_rotation(chest,root@cv2.Rodrigues(residual)[0],profile)
            self.last_good['chest']=(t,root.T@chest)
        elif profile>0:
            local = self.last_good.get('chest',(t,I))[1]
            chest = blend_rotation(chest,root@local,profile)
        rots['spine3'] = root.T@chest
        info['heading_deg'] = math.degrees(self.heading)
        info['profile_weight'] = profile
        info['solver_version'] = SOLVER_VERSION
        for side, hip, knee, ankle, shoulder, elbow, wrist, direction in (
                ('left', 23, 25, 27, 11, 13, 15, [1, 0, 0]),
                ('right', 24, 26, 28, 12, 14, 16, [-1, 0, 0])):
            good = q is not None and visible([hip, knee, ankle], .55)
            info[side+'_leg'] = good
            if good:
                thigh = rotation([0, -1, 0], root.T@(q[knee]-q[hip]))
                shin = rotation([0, -1, 0], root.T@(q[ankle]-q[knee]))
                leg = (thigh, thigh.T @ shin, shin.T)
                self.last_good[side+'_leg'] = (t, leg)
            elif getattr(self,'best_stability',False) and q is not None and visible([hip,knee],.55):
                # A hidden ankle must not freeze a visible moving thigh or snap
                # the entire leg upright. Keep the last knee bend in leg-local
                # space while the observed hip/knee continue to move.
                thigh=rotation([0,-1,0],root.T@(q[knee]-q[hip]))
                previous=self.last_good.get(side+'_leg',(t,(I,I,I)))[1]
                bend=previous[1]
                leg=(thigh,bend,(thigh@bend).T)
                self.last_good[side+'_leg']=(t,leg)
            elif side+'_leg' in self.last_good and (getattr(self,'best_stability',False) or t-self.last_good[side+'_leg'][0] < .20):
                leg = self.last_good[side+'_leg'][1]
            else:
                leg = (I, I, I)
            for name, R in zip(('hip', 'knee', 'ankle'), leg):
                rots[side+'_'+name] = R
            arm_good = q is not None and visible([shoulder, elbow, wrist], .5)
            info[side+'_arm'] = arm_good
            if arm_good:
                upper = root@rotation(direction, root.T@(q[elbow]-q[shoulder]))
                fore = root@rotation(direction, root.T@(q[wrist]-q[elbow]))
                orientation = solve_forearm_roll(raw.get(side+'_hand', []), side, fore)
                if orientation is not None:
                    self.last_good[side+'_orientation'] = (t, orientation)
                elif side+'_orientation' in self.last_good and t-self.last_good[side+'_orientation'][0]<.4:
                    orientation = self.last_good[side+'_orientation'][1]
                else:
                    orientation = (I, I)
                twist, wrist_bend = orientation
                if getattr(self,'full_sleeves',False):
                    # Cloth sleeve roll is unobservable from an intermittently
                    # detected palm. Do not spin the printed forearm when the
                    # hand appears, vanishes or changes its ambiguous normal.
                    # Keep observed shoulder/elbow/wrist swing; carry the palm
                    # orientation on the hand bone alone (same global hand).
                    wrist_bend=twist@wrist_bend
                    twist=I
                arm = (chest.T@upper, upper.T@fore@twist, wrist_bend)
                self.last_good[side+'_arm'] = (t,arm)
            elif (getattr(self,'male_livefit',False) or (getattr(self,'female_livefit',False) and getattr(self,'best_stability',False))) and q is not None and visible([shoulder,elbow],.5):
                # Hidden wrist must not lower an OBSERVED moving upper arm.
                upper=root@rotation(direction,root.T@(q[elbow]-q[shoulder]))
                old_stamp,previous=self.last_good.get(side+'_arm',(t-2,(I,I,I)))
                bend=blend_rotation(previous[1],I,ramp(t-old_stamp,.2,.65))
                arm=(chest.T@upper,bend,previous[2] if t-old_stamp<.2 else I)
            else:
                # A missing far arm is not a measured T pose. Hold its last
                # BODY-LOCAL pose, then gradually lower it beside the torso.
                sx = direction[0]
                relaxed = (rotation(direction,[sx*.18,-1,.06]),I,I)
                stamp, old = self.last_good.get(side+'_arm',(t-2,relaxed))
                weight = ramp(t-stamp,.4,1.2)
                arm = tuple(blend_rotation(a,b,weight) for a,b in zip(old,relaxed))
            for name,R in zip(('shoulder','elbow','wrist'),arm):
                rots[side+'_'+name]=R
        face = np.asarray(raw.get('face', []), dtype=float)
        head = I.copy()
        if len(face) >= 468:
            f = face[:, :3].copy(); f[:, 0] *= aspect; f[:, 1:] *= -1; f[:, 2] *= aspect
            # 33 lies on the anatomical right; 263 on the anatomical left.
            measured = frame(f[263]-f[33], f[10]-f[152])
            if self.head_zero is None:
                self.head_samples.append(measured)
                if len(self.head_samples) >= 20:
                    recent = self.head_samples[-20:]
                    average = mean_rotation(recent)
                    spread = max(np.linalg.norm(rv(average.T @ a)) for a in recent)
                    if spread < math.radians(12):
                        self.head_zero = average
            if self.head_zero is not None:
                v = np.clip(rv(measured @ self.head_zero.T), [-.65, -1.65, -.55], [.65, 1.65, .55])
                head = cv2.Rodrigues(v)[0]
                self.last_good['head'] = (t, head, chest.T@head)
        elif 'head' in self.last_good:
            stamp, old, relative = self.last_good['head']
            # Preserve the calibrated pitch, but follow a profile body turn
            # when the face is no longer observable from a single camera.
            fallback = chest@blend_rotation(relative,I,ramp(t-stamp,.25,.8))
            front = old if t-stamp<.25 else I
            head = blend_rotation(front,fallback,profile)
        else:
            head = blend_rotation(head,chest,profile)
        rots['head'] = chest.T @ head
        info['head_calibrated'] = self.head_zero is not None
        info['calibration_samples'] = min(len(self.head_samples), 20)
        body = [rv(self.smooth(n, rots[n], dt)).tolist() for n in BODY_JOINT_NAMES]
        hands = []
        for side in ('left', 'right'):
            points = raw.get(side+'_hand', [])
            good = len(points) == 21
            info[side+'_hand'] = good
            if good:
                curls = solve_hand(points, side)
                self.last_good[side+'_hand'] = (t, curls)
            elif side+'_hand' in self.last_good and t-self.last_good[side+'_hand'][0] < .25:
                curls = self.last_good[side+'_hand'][1]
            else:
                curls = np.zeros((15, 3))
            hands.append([rv(self.smooth(side+'_finger'+str(i), cv2.Rodrigues(c)[0], dt)).tolist() for i,c in enumerate(curls)])
        if getattr(self,'female_livefit',False) and getattr(self,'best_stability',False):
            # HeadingFollower already filters this rotation. Filtering pelvis
            # again makes the rendered root lag behind the very root used to
            # compensate the local sleeves/chest in this same packet.
            pelvis=root;self.filtered['pelvis']=root.copy()
        else:pelvis=self.smooth('pelvis',root,dt)
        payload = PosePayload(rv(pelvis).tolist(), body, hands[0], hands[1], ZERO.copy(), [0.]*10, [0.]*10, ZERO.copy())
        return payload, info

def points(landmarks):
    return [[float(p.x), float(p.y), float(p.z), float(p.visibility or 0)] for p in landmarks]

def camera_frames(args):
    import mediapipe as mp
    opt = mp.tasks.vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_pose_detection_confidence=.5, min_pose_landmarks_confidence=.5, min_hand_landmarks_confidence=.4)
    detector = mp.tasks.vision.HolisticLandmarker.create_from_options(opt)
    source = args.camera_index if args.source is None else (int(args.source) if args.source.isdecimal() else args.source)
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW if isinstance(source, int) else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        detector.close(); raise RuntimeError('Camera unavailable. Stop other camera applications, then retry.')
    solver = Solver(); start = time.monotonic(); seq = 0; stamp = 0; recording = []
    phases = [('SETUP: whole body + both feet visible', 10), ('STAND STRAIGHT - look forward', 7),
              ('Lift ONLY LEFT leg, lower, repeat', 10), ('Lift ONLY RIGHT leg, lower, repeat', 10),
              ('Move hands closer: OPEN - FIST - OPEN', 15)]
    if args.profile_test:
        phases = [('SETUP: step back; full body visible', 10),
                  ('FRONT: stand straight; arms relaxed', 8),
                  ('Turn slowly LEFT to SIDE; hold; bend arm', 14),
                  ('Return FRONT - stand straight', 6),
                  ('Turn slowly RIGHT to SIDE; hold; bend arm', 14),
                  ('FRONT: lift LEFT leg, then RIGHT leg', 10),
                  ('Hands closer: palms UP/DOWN, OPEN/FIST', 12)]
    phase = -1; profile_ready = False; ready_count = 0; profile_start = 0.
    try:
        while True:
            if args.duration and time.monotonic()-start > args.duration: break
            ok, bgr = cap.read(); captured = time.time()
            if not ok:
                if args.source_loop and isinstance(source, str): cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                break
            elapsed = time.monotonic()-start
            label = 'Keep face straight for calibration; C = recalibrate; ESC = stop'
            if args.profile_test and not profile_ready:
                label = f'STEP BACK: one person, head + BOTH feet visible ({ready_count}/12)'
            elif args.test_sequence or args.profile_test:
                phase_elapsed = elapsed-profile_start if args.profile_test else elapsed
                end = 0; new_phase = -1
                for k, (msg, seconds) in enumerate(phases):
                    end += seconds
                    if phase_elapsed < end: new_phase = k; label = f'{msg} ({int(end-phase_elapsed)}s)'; break
                if new_phase < 0: break
                if new_phase != phase:
                    phase = new_phase
                    if phase == 1: solver.reset_calibration()
                    print('TEST_PHASE '+str(phase)+' '+label, file=__import__('sys').stderr, flush=True)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            stamp = max(stamp+1, int(elapsed*1000))
            r = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), stamp)
            raw = {'t': elapsed, 'phase': phase, 'pose': points(r.pose_landmarks), 'world': points(r.pose_world_landmarks),
                   'left_hand': points(r.left_hand_world_landmarks), 'right_hand': points(r.right_hand_world_landmarks), 'face': points(r.face_landmarks)}
            payload, info = solver.solve(raw, bgr.shape[1]/bgr.shape[0])
            if args.profile_test and not profile_ready:
                ready_count = ready_count+1 if info['left_leg'] and info['right_leg'] and len(raw['face'])>=468 else 0
                if ready_count>=12:
                    profile_ready = True; profile_start = elapsed-10
                    solver.reset_calibration()
            out = PoseFrame(SCHEMA_VERSION, seq, captured, 'ok', payload); seq += 1
            if args.record_landmarks: recording.append({'raw': raw, 'info': info, 'frame': asdict(out)})
            if args.preview_window or args.test_sequence or args.profile_test:
                view = cv2.flip(bgr, 1)
                for a,b in [(11,13),(13,15),(12,14),(14,16),(23,25),(25,27),(24,26),(26,28)]:
                    if len(raw['pose']) == 33:
                        pa,pb = raw['pose'][a],raw['pose'][b]
                        if min(pa[3],pb[3]) > .5:
                            xy=lambda p: (int((1-p[0])*view.shape[1]),int(p[1]*view.shape[0]))
                            cv2.line(view,xy(pa),xy(pb),(50,210,90),3)
                cv2.rectangle(view,(0,0),(view.shape[1],115),(20,20,20),-1)
                cv2.putText(view,label,(14,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,240,255),2)
                flags = '  '.join(k+':'+('OK' if info[k] else 'NOT SEEN') for k in ['left_leg','right_leg','left_hand','right_hand'])
                cv2.putText(view,flags,(14,65),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
                cal = 'HEAD calibrated' if info['head_calibrated'] else f"HEAD: hold still {info['calibration_samples']}/20"
                cv2.putText(view,cal+f" | BODY turn {info.get('heading_deg',0):+.0f} deg | "+SOLVER_VERSION,(14,95),cv2.FONT_HERSHEY_SIMPLEX,.55,(100,255,100),1)
                cv2.imshow('Fahad Live - local camera',view)
                key = cv2.waitKey(1)&255
                if key == 27: break
                if key in (ord('c'),ord('C')): solver.reset_calibration()
            yield out
    finally:
        cap.release(); detector.close(); cv2.destroyAllWindows()
        if args.record_landmarks:
            Path(args.record_landmarks).write_text(json.dumps(recording), encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['live','capture'])
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--host', default='127.0.0.1'); parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--camera-index', type=int, default=0)
    parser.add_argument('--source'); parser.add_argument('--source-loop', action='store_true')
    parser.add_argument('--width', type=int, default=1280); parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--parent-pid', type=int); parser.add_argument('--log-file')
    parser.add_argument('--preview-window', action='store_true')
    parser.add_argument('--yolo-model'); parser.add_argument('--yolo-threshold')
    parser.add_argument('--duration', type=float, default=0)
    parser.add_argument('--test-sequence', action='store_true'); parser.add_argument('--record-landmarks')
    parser.add_argument('--profile-test', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(filename=args.log_file, level=logging.INFO)
    stream = camera_frames(args)
    try:
        if args.mode == 'capture':
            for _ in stream: pass
        else:
            from posecap_engine.stream_server import serve_once
            from posecap_engine.watchdog import ParentWatchdog
            serve_once(stream, host=args.host, port=args.port, watchdog=ParentWatchdog(args.parent_pid),
                       ready=lambda address: print(json.dumps({'event':'listening','host':address[0],'port':address[1]}),flush=True))
    finally:
        stream.close()

if __name__ == '__main__':
    main()
