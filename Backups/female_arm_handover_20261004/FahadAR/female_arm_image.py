"""Current-image arm motion with bounded detector-innovation damping.

Not pose extrapolation. No change to quiet-region locks, confidence or depth.
Two independently supported sleeve segments must agree on their shared elbow.
Missing/ambiguous imagery, depth changes and large moves keep the old path.
"""
import numpy as np
from female_reliability import fresh


class ArmImageFollower:
    def __init__(self):self.reset()
    def reset(self):
        self.last=None;self.pose=None;self.world=None;self.maps={};self.state={}

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

    def update(self,pose,world,stamp,motion):
        p=np.asarray(pose,float).copy();w=np.asarray(world,float)
        if p.shape!=(33,4) or w.shape!=(33,4) or not np.isfinite(p).all() or not np.isfinite(w).all() or not fresh(motion,stamp):
            self.reset();return pose,world
        current={g:self.transform(motion.get('regions',{}).get(str(g),{})) for g in (2,3,4,5)}
        continuous=self.last is not None and 0<stamp-self.last<=.30
        self.state={}
        for name,idx,gs in [('left',[11,13,15],[2,3]),('right',[12,14,16],[4,5])]:
            state={'corrected':False,'reason':'unobserved'};self.state[name]=state
            regions=[motion.get('regions',{}).get(str(g),{}) for g in gs]
            # Preserve settled stillness exactly, including its fixed anchors.
            if all(e.get('quiet') for e in regions):state['reason']='existing_stillness';continue
            if not continuous or any(current[g] is None for g in gs):continue
            if min(p[idx,3])<.55 or min(self.pose[idx,3])<.55:continue
            if not np.all((p[idx,:2]>.01)&(p[idx,:2]<.99)):continue
            steps=[]
            for g in gs:
                ref,anchor,m=current[g];old=self.maps.get(g)
                if old is not None and old[:2]==(ref,anchor):step=m@np.linalg.inv(old[2])
                elif abs(anchor-self.last)<.001:step=m
                else:break
                scale=np.sqrt(np.linalg.det(step));angle=abs(np.arctan2(step[1,0],step[0,0]))
                if not .85<scale<1.18 or angle>np.deg2rad(30):break
                steps.append(step)
            if len(steps)!=2:continue
            prior=self.pose[idx,:2]*[640,360]
            upper=prior[:2]@steps[0][:2,:2].T+steps[0][:2,2]
            fore=prior[1:]@steps[1][:2,:2].T+steps[1][:2,2]
            if np.linalg.norm(upper[1]-fore[0])>1.25:state['reason']='segments_disagree';continue
            if np.linalg.norm(upper[0]-p[idx[0],:2]*[640,360])>2.5:state['reason']='shoulder_disagrees';continue
            depth=(w[idx,:3]-w[idx[0],:3])-(self.world[idx,:3]-self.world[idx[0],:3])
            if np.linalg.norm(depth,axis=1).max()>.08:state['reason']='depth_changed';continue
            predicted=np.array([(upper[1]+fore[0])*.5,fore[1]])
            if np.linalg.norm(predicted-prior[1:],axis=1).max()>35:state['reason']='rapid_motion';continue
            measured=p[idx[1:],:2]*[640,360];innovation=measured-predicted
            error=np.linalg.norm(innovation,axis=1)
            if error.max()>12:state['reason']='large_innovation';continue
            # Clean detector/image agreement is bit-for-bit unchanged. For
            # conflicting observations use measured image motion immediately,
            # damping only the innovation, NOT the arm's actual displacement.
            changed=error>1.25
            corrected=predicted+.35*innovation
            for j,xy,use in zip(idx[1:],corrected,changed):
                if use:p[j,:2]=xy/[640,360]
            state.update(corrected=bool(changed.any()),reason='image_motion_fusion' if changed.any() else 'agrees',
                         innovation_px=error.tolist())
        self.last=stamp;self.pose=p.copy();self.world=w.copy();self.maps=current
        return p.tolist(),world
