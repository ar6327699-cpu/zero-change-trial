"""Repair image-contradicted sleeve observations BEFORE temporal filtering.

Female best-stability only. Still data and genuine articulation retain the
existing path. A translated sleeve must not inherit a held torso's deadband;
the repaired observation passes through exactly the same downstream filters.
"""
import numpy as np
from female_reliability import RegionalLock, GROUPS, fresh


class ArmTranslationRepair(RegionalLock):
    def update(self, pose, world, stamp, motion):
        p=np.asarray(pose,float).copy();w=np.asarray(world,float).copy()
        self.state={}
        if (p.shape!=(33,4) or w.shape!=(33,4) or not np.isfinite(p).all() or
                not np.isfinite(w).all() or not np.isfinite(stamp) or not fresh(motion,stamp)):
            self.reset();return pose,world
        if self.last is not None and not 0<stamp-self.last<=.6:self.reset()
        self.last=stamp
        regions=motion.get('regions',{})
        for name,idx in [('left_arm',[11,13,15]),('right_arm',[12,14,16])]:
            evidence=[regions.get(str(g),{}) for g in GROUPS[name]]
            usable=all(e.get('valid') and e.get('shape_preserved') for e in evidence)
            usable=usable and np.all(p[idx,3]>=.45) and np.all((p[idx,:2]>.005)&(p[idx,:2]<.995))
            optical=[(e.get('position_epoch'),np.asarray(e.get('position_px',[np.nan,np.nan]),float)) for e in evidence]
            usable=usable and all(v.shape==(2,) and np.isfinite(v).all() for _,v in optical)
            quiet=all(e.get('quiet') for e in evidence)
            old=self.anchors.get(name)
            reason='unobserved_or_articulating'
            if usable and old is not None:
                pp,ww,prior=old
                same=all(a[0]==b[0] and a[0] is not None for a,b in zip(optical,prior))
                shifts=np.array([a[1]-b[1] for a,b in zip(optical,prior)])
                common=shifts.mean(0)
                shoulder=(p[idx[0],:2]-pp[0,:2])*[640,360]
                follows=same and np.linalg.norm(shifts-common,axis=1).max()<=.6 and np.linalg.norm(shoulder-common)<=2.5
                if follows:
                    if quiet and np.linalg.norm(common)<=1.25:
                        # Leave successful stillness behavior entirely alone.
                        self.pending.pop(name,None)
                        self.state[name]={'repaired':False,'reason':'existing_stillness'}
                        continue
                    delta=((p[idx,:2]-p[idx[0],:2])-(pp[:,:2]-pp[0,:2]))*[640,360]
                    depth=(w[idx,:3]-w[idx[0],:3])-(ww[:,:3]-ww[0,:3])
                    pixel_errors=np.linalg.norm(delta,axis=1)
                    pixels=float(pixel_errors.max());dz=float(np.linalg.norm(depth,axis=1).max())
                    # A single displaced elbow/wrist with consistent depth
                    # may be articulation at a poorly sampled sleeve edge.
                    # Do not make that real limb wait for pose consensus.
                    conflict=np.count_nonzero(pixel_errors[1:]>6)>=2 or dz>.18
                    if pixels>6 and not conflict:
                        self.anchors.pop(name,None);self.pending.pop(name,None)
                        self.state[name]={'repaired':False,'reason':'single_joint_articulation'}
                        continue
                    bounded=pixels<=18 and dz<=.35
                    if not conflict:
                        self.pending.pop(name,None)
                        self.state[name]={'repaired':False,'reason':'clean_translation'}
                        continue
                    if bounded and self._unconfirmed_arm_jump(name,delta,depth,stamp):
                        # Optical displacement corroborates common movement;
                        # the CURRENT raw shoulder supplies translation. Do
                        # not integrate subpixel flow into rendered position,
                        # or inherit the later torso filter's paused shoulder.
                        p[idx[1:],:3]=pp[1:,:3]+p[idx[0],:3]-pp[0,:3]
                        w[idx[1:],:3]=ww[1:,:3]+w[idx[0],:3]-ww[0,:3]
                        self.state[name]={'repaired':True,'reason':'image_rejected_inconsistent_pose',
                                          'pixel_max':pixels,'depth_max':dz}
                        continue
                    reason='confirmed_innovation' if bounded else 'large_innovation'
                else:reason='articulation_or_reference_changed'
            self.anchors.pop(name,None);self.pending.pop(name,None)
            if usable and quiet:
                self.anchors[name]=(p[idx].copy(),w[idx].copy(),optical)
            self.state[name]={'repaired':False,'reason':reason}
        return p.tolist(),w.tolist()
