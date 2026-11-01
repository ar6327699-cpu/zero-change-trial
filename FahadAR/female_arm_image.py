"""Independent image-supported arm observations and continuous motion handover.

No future prediction, inferred visibility or depth correction. A reliable
segment survives a weak neighbour. A short movement latch avoids re-freezing
on every renewed optical reference; truly stationary evidence still uses the
existing settled lock. The caller commits its final filtered/locked input so
the next image step starts from the pose actually handed to the solver.
"""
import numpy as np
from female_reliability import fresh


class ArmImageFollower:
    def __init__(self):self.reset()
    def reset(self):
        self.last=None;self.pose=None;self.world=None;self.maps={};self.state={}
        self.phases={};self.moving=[];self.raw=None;self.followed={}

    @staticmethod
    def transform(region):
        e=region.get('image_transform',{})
        try:
            m=np.asarray(e.get('matrix'),float)
            if not region.get('valid') or m.shape!=(2,3) or not np.isfinite(m).all():return None
            span=float(e.get('span_ratio',0));residual=float(e.get('residual_px',1));stamp=float(e['stamp'])
            if not np.isfinite([span,residual,stamp]).all() or span<.40 or residual>.5:return None
            h=np.eye(3);h[:2]=m
            if not .5<np.linalg.det(h)<2:return None
            return region.get('reference'),stamp,h
        except (TypeError,ValueError,KeyError):return None

    def _step(self,g,current):
        item=current[g]
        if item is None:return None
        ref,anchor,m=item;old=self.maps.get(g)
        if old is not None and old[:2]==(ref,anchor):step=m@np.linalg.inv(old[2])
        elif abs(anchor-self.last)<.001:step=m
        else:return None
        scale=np.sqrt(np.linalg.det(step));angle=abs(np.arctan2(step[1,0],step[0,0]))
        return step if .85<scale<1.18 and angle<=np.deg2rad(30) else None

    def _motion(self,name,delta,stamp,released=False):
        """Evidence latch, NOT output smoothing or an extended frozen pose.

        Coherent subpixel steps accumulate against a fixed articulation
        anchor. Reversing affine-fit noise cannot repeatedly renew motion
        merely because each individual step is nonzero. Larger articulation
        responds immediately; image-reference renewal preserves this state.
        """
        old=self.phases.get(name,{})
        if delta is None:
            self.phases.pop(name,None);return False
        delta=np.asarray(delta,float)
        if not np.isfinite(delta).all():
            self.phases.pop(name,None);return False
        size=float(np.linalg.norm(delta.reshape(-1,2),axis=1).max())
        position=old.get('position',np.zeros_like(delta))+delta
        anchor=old.get('anchor',np.zeros_like(delta))
        travel=float(np.linalg.norm((position-anchor).reshape(-1,2),axis=1).max())
        prior=old.get('delta')
        coherent=(prior is not None and size>.05 and
                  np.dot(delta,prior)>.5*np.linalg.norm(delta)*np.linalg.norm(prior))
        count=old.get('count',0)+1 if coherent else (1 if size>.05 else 0)
        # Checkpoint only confirmed net travel, never every noisy step.
        # Two 0.4px steps follow promptly; alternating +/-0.4px stays within
        # the checkpoint and settles even after a genuine move activated it.
        active=old.get('active',False);last_move=old.get('last_move',stamp)
        threshold=.35 if active else .65
        confirmed=(released and size>1.25) or (travel>threshold and count>=2)
        if confirmed:
            active=True;last_move=stamp;anchor=position.copy()
        elif active and stamp-last_move>=.40:
            active=False
        self.phases[name]=dict(active=active,last_move=last_move,delta=delta.copy(),
                               count=count,position=position.copy(),anchor=anchor.copy())
        return active

    def motion_evidence(self,motion):
        """Only moving arms override quiet votes; other regions stay intact."""
        if not self.moving:return motion
        result=dict(motion);result['regions']=dict(motion.get('regions',{}))
        result['arm_follow_joints']={}
        for name,gs in [('left',(2,3)),('right',(4,5))]:
            if name not in self.moving:continue
            supported=self.state[name].get('supported_segments',[])
            elbow,wrist=(13,15) if name=='left' else (14,16)
            result['arm_follow_joints'][name]=([elbow] if supported else [])+([wrist] if gs[1] in supported else [])
            for g in gs:
                r=dict(result['regions'].get(str(g),{}))
                r['image_quiet']=r.get('quiet',False);r['quiet']=False
                # A cumulative rigid vote can still be true on a reversing
                # arm. Preserve it as diagnostic evidence, but the current
                # verified articulation must win every downstream hold vote.
                r['image_rigid_translation']=r.get('rigid_translation',False)
                r['rigid_translation']=False
                result['regions'][str(g)]=r
        result.update(quiet=False,mode='moving',reason='arm_motion_continuity',arm_follow_active=list(self.moving))
        return result

    def commit(self,pose,world,stamp):
        """Synchronize the image predictor with downstream locks, not raw noise."""
        p=np.asarray(pose,float);w=np.asarray(world,float)
        if self.last==stamp and p.shape==w.shape==(33,4) and np.isfinite(p).all() and np.isfinite(w).all():
            self.pose=p.copy();self.world=w.copy()

    def retain_motion(self,pose,world,body_lock,sleeve_lock):
        """Do not quantize confirmed optical movement with two quiet XY locks.

        Only supported distal XY is passed through, shoulder-relative to the
        filtered torso. All depth and confidence filters stay in force. Keep
        hidden filter states aligned so returning to quiet cannot spring back.
        """
        if not self.moving:return pose,world
        p=np.asarray(pose,float).copy();w=np.asarray(world,float)
        for name,idx,gs in [('left',[11,13,15],[2,3]),('right',[12,14,16],[4,5])]:
            if name not in self.moving:continue
            supported=self.state[name].get('supported_segments',[])
            for joint in ([idx[1],idx[2]] if gs[1] in supported else [idx[1]]):
                p[joint,:2]=self.pose[joint,:2]-self.pose[idx[0],:2]+p[idx[0],:2]
                if body_lock.previous is not None:body_lock.previous[joint,:2]=p[joint,:2]
            sleeve_lock.anchors[tuple(idx)]=(p[idx].copy(),w[idx].copy())
        return p.tolist(),world

    def update(self,pose,world,stamp,motion):
        p=np.asarray(pose,float).copy();w=np.asarray(world,float)
        if p.shape!=(33,4) or w.shape!=(33,4) or not np.isfinite(p).all() or not np.isfinite(w).all() or not fresh(motion,stamp):
            self.reset();return pose,world
        current={g:self.transform(motion.get('regions',{}).get(str(g),{})) for g in (2,3,4,5)}
        continuous=self.last is not None and 0<stamp-self.last<=.30
        self.state={};self.moving=[]
        previous_followed=self.followed;self.followed={}
        for name,idx,gs in [('left',[11,13,15],[2,3]),('right',[12,14,16],[4,5])]:
            state={'corrected':False,'reason':'unobserved'};self.state[name]=state
            regions=[motion.get('regions',{}).get(str(g),{}) for g in gs]
            quiet=all(e.get('valid') and e.get('quiet') for e in regions)
            if not continuous:
                self.phases.pop(name,None)
                if quiet:state['reason']='existing_stillness'
                continue
            steps=[self._step(g,current) for g in gs]
            for k,part in enumerate((idx[:2],idx[1:])):
                if min(p[part,3])<.55 or min(self.pose[part,3])<.55 or not np.all((p[part,:2]>.01)&(p[part,:2]<.99)):
                    steps[k]=None
            if min(p[idx[0],3],self.pose[idx[0],3])<.55:
                steps=[None,None]
            state['supported_segments']=[g for g,s in zip(gs,steps) if s is not None]
            state['segment_span']=[e.get('image_transform',{}).get('span_ratio') for e in regions]
            prior=self.pose[idx,:2]*[640,360]
            upper=None if steps[0] is None else prior[:2]@steps[0][:2,:2].T+steps[0][:2,2]
            fore=None if steps[1] is None else prior[1:]@steps[1][:2,:2].T+steps[1][:2,2]
            if upper is None and fore is None:
                self.phases.pop(name,None)
                if quiet:state['reason']='existing_stillness'
                continue
            if upper is not None and fore is not None and np.linalg.norm(upper[1]-fore[0])>1.25:
                self.phases.pop(name,None);state['reason']='segments_disagree';continue
            if upper is not None:
                # Compare like with like: the image-measured shoulder STEP vs
                # the detector's own shoulder step. prior is the committed,
                # torso-locked pose; comparing it with the RAW shoulder also
                # counted the lock offset and vetoed genuine sleeve motion.
                raw_step=(p[idx[0],:2]-self.raw[idx[0],:2])*[640,360] if self.raw is not None else p[idx[0],:2]*[640,360]-prior[0]
                state['shoulder_px']=float(np.linalg.norm(upper[0]-prior[0]-raw_step))
                if state['shoulder_px']>2.5:
                    self.phases.pop(name,None);state['reason']='shoulder_disagrees';continue
            depth=(w[idx,:3]-w[idx[0],:3])-(self.world[idx,:3]-self.world[idx[0],:3])
            depth_indices=[0,1,2] if fore is not None else [0,1]
            depth_changed=np.linalg.norm(depth[depth_indices],axis=1).max()>.08
            if depth_changed and name!='left':
                self.phases.pop(name,None);state['reason']='depth_changed';continue
            supported=np.array([True,fore is not None])
            elbow=(upper[1]+fore[0])*.5 if upper is not None and fore is not None else upper[1] if upper is not None else fore[0]
            predicted=np.array([elbow,fore[1] if fore is not None else p[idx[2],:2]*[640,360]])
            if np.linalg.norm((predicted-prior[1:])[supported],axis=1).max()>35:
                self.phases.pop(name,None);state['reason']='rapid_motion';continue
            measured=p[idx[1:],:2]*[640,360];innovation=measured-predicted
            error=np.linalg.norm(innovation,axis=1)
            if depth_changed:
                # A monocular depth jump need not invalidate independently
                # corroborated LEFT image-plane motion. Keep the original
                # veto unless both segment fits exist and current detector XY
                # agrees within 3px. This never replaces or promotes depth,
                # visibility, or the right arm's established response.
                if upper is None or fore is None or error.max()>3.:
                    self.phases.pop(name,None);state['reason']='depth_changed';continue
                state['depth_guard']='current_xy_corroborated'
            if error[supported].max()>12:
                self.phases.pop(name,None);state['reason']='large_innovation';continue
            # The relative swing of each supported segment, independent of
            # torso translation. A missing neighbour is not zero-motion proof.
            swing=np.zeros((2,2))
            if upper is not None:swing[0]=(upper[1]-upper[0])-(prior[1]-prior[0])
            if fore is not None:swing[1]=(fore[1]-fore[0])-(prior[2]-prior[1])
            moving=self._motion(name,swing.ravel(),stamp,released=not quiet)
            state['motion_active']=moving
            if moving:self.moving.append(name)
            prior_support=previous_followed.get(name,())
            handover=supported&np.array([bool(prior_support),gs[1] in prior_support])
            if moving:self.followed[name]=state['supported_segments']
            elif prior_support:self.followed[name]=[g for g in state['supported_segments'] if g in prior_support]
            if quiet and not moving:
                if handover.any():
                    # The detector/fused offset must not suddenly disappear
                    # when a moving sleeve settles. Carry the last committed
                    # position by THIS frame's verified optical step. Existing
                    # regional locks then acquire that same position, rather
                    # than a fresh raw-detector jump. Missing evidence, depth
                    # changes or a discontinuity already reset this handover.
                    for j,xy,use in zip(idx[1:],predicted,handover):
                        if use:p[j,:2]=xy/[640,360]
                    state.update(corrected=True,reason='quiet_motion_handover',
                                 innovation_px=error.tolist(),corrected_joints=[j for j,use in zip(idx[1:],handover) if use])
                else:state['reason']='existing_stillness'
                continue
            # Clean detector/image agreement is bit-for-bit unchanged. For
            # conflicting observations use measured image motion immediately,
            # damping only the innovation, NOT the arm's actual displacement.
            changed=(error>1.25)&supported
            corrected=predicted+.35*innovation
            for j,xy,use in zip(idx[1:],corrected,changed):
                if use:p[j,:2]=xy/[640,360]
            state.update(corrected=bool(changed.any()),reason=('image_motion_fusion' if supported.all() else 'partial_image_fusion') if changed.any() else 'agrees',
                         innovation_px=error.tolist(),corrected_joints=[j for j,use in zip(idx[1:],changed) if use])
        self.last=stamp;self.pose=p.copy();self.world=w.copy();self.maps=current
        self.raw=np.asarray(pose,float).copy()
        return p.tolist(),world
