"""Keep corroborated quiet female regions quiet all the way to rendered geometry.

Landmarks alone are not the final output: solver filters, confidence-weighted
projection, and residual fitting can still move a held torso. This guard keeps
their committed values together. It never changes mesh data or fitting rules.
Unknown/moving evidence takes the existing live path without an added delay.
"""
import numpy as np
from female_reliability import GROUPS, REGIONS

BONES={'torso':('Hips','Spine','Spine1','Spine2'),
       'left_arm':('LeftShoulder','LeftArm','LeftForeArm','LeftHand'),
       'right_arm':('RightShoulder','RightArm','RightForeArm','RightHand'),
       'left_leg':('LeftUpLeg','LeftLeg','LeftFoot','LeftToeBase','LeftToe_End'),
       'right_leg':('RightUpLeg','RightLeg','RightFoot','RightToeBase','RightToe_End')}

class RenderLock:
    def __init__(self):self.reset()
    def reset(self):
        self.last=None;self.context=None;self.tokens={};self.held=[]
        self.bones={};self.offsets=[];self.values={};self.fit=None;self.matrix=None

    def begin(self,meta,context,arm,garments,fit):
        """Snapshot the previous committed frame, not an unseen worker frame."""
        stamp=float(meta['captured_at']);motion=meta.get('female_motion') or {}
        state=meta.get('female_region_state') or {}
        prior=self.tokens
        continuous=self.context==context and self.last is not None and 0<stamp-self.last<=.6
        self.held=[];self.bones={};self.offsets=[];self.values={};self.fit=None;self.matrix=None
        fresh=abs(float(motion.get('stamp',-1))-stamp)<.001
        evidence=motion.get('regions',{})
        if continuous and fresh and isinstance(evidence,dict) and fit is not None:
            for name,groups in GROUPS.items():
                r=state.get(name,{})
                # Tokens must match a frame Blender actually consumed. If the
                # queue skipped a movement/re-anchor, never hold the old pose.
                token=r.get('anchor')
                if r.get('held') and token is not None and prior.get(name)==token:
                    if all(evidence.get(str(g),{}).get('valid') and evidence.get(str(g),{}).get('quiet') for g in groups):
                        self.held.append(name)
        self.context=context;self.last=stamp
        self.tokens={name:r.get('anchor') for name,r in state.items() if r.get('anchor') is not None}
        # A changing body transform needs live parent-relative corrections.
        # Preserve the established movement path unless the torso is held.
        if 'torso' not in self.held:self.held=[];return
        names={b for r in self.held for b in BONES[r]}
        for b in arm.pose.bones:
            name=b.name.removeprefix('mixamorig:')
            fingers=any(r in self.held and name.startswith(prefix+'Hand') for r,prefix in (('left_arm','Left'),('right_arm','Right')))
            if name in names or fingers:
                # Restoring matrix_basis repeatedly decomposes/re-normalizes
                # rotations in Blender and itself creates tiny numeric creep.
                self.bones[b.name]=(b.rotation_mode,tuple(b.location),tuple(b.scale),
                                    tuple(b.rotation_quaternion),tuple(b.rotation_euler),tuple(b.rotation_axis_angle))
        self.fit=np.asarray(fit,float).copy();self.matrix=arm.matrix_world.copy()
        ids=set(REGIONS['torso'])
        for r in self.held:
            if r!='torso':ids.update(REGIONS[r][1:])
        for obj,mod,sockets in garments.items.values():
            self.offsets.append((obj,mod,{key:tuple(mod[key]) for i,key in sockets.items() if i in ids}))
        self.values={i:v.copy() for i,v in garments.female_lock.values.items() if i in ids}

    def restore_pose(self,arm):
        # A solver expresses moving limbs relative to its latest chest/root.
        # When the quiet parent is restored, retain the moving limb's target
        # direction, not that now-outdated parent-relative rotation.
        matrices={}
        def pose(b):
            if b.name not in matrices:
                kw={} if b.parent is None else dict(parent_matrix=pose(b.parent),parent_matrix_local=b.parent.bone.matrix_local)
                matrices[b.name]=b.bone.convert_local_to_pose(b.matrix_basis,b.bone.matrix_local,**kw)
            return matrices[b.name]
        moving={}
        if self.bones:
            for region,name in (('left_arm','LeftArm'),('right_arm','RightArm'),('left_leg','LeftUpLeg'),('right_leg','RightUpLeg')):
                if region not in self.held:
                    b=arm.pose.bones['mixamorig:'+name];moving[b.name]=pose(b).copy()
        for name,(mode,loc,scale,quat,euler,axis) in self.bones.items():
            b=arm.pose.bones[name];b.rotation_mode=mode;b.location=loc;b.scale=scale
            b.rotation_quaternion=quat;b.rotation_euler=euler;b.rotation_axis_angle=axis
        matrices.clear()
        for name,target in moving.items():
            b=arm.pose.bones[name];parent=pose(b.parent)
            local=b.bone.convert_local_to_pose(target,b.bone.matrix_local,
                parent_matrix=parent,parent_matrix_local=b.parent.bone.matrix_local,invert=True)
            # Preserve the connected joint's position and authored scale.
            # Only compensate its local orientation for the restored parent.
            q=local.to_quaternion()
            if b.rotation_mode=='QUATERNION':b.rotation_quaternion=q
            elif b.rotation_mode=='AXIS_ANGLE':
                axis,angle=q.to_axis_angle();b.rotation_axis_angle=(angle,*axis)
            else:b.rotation_euler=q.to_euler(b.rotation_mode)
            matrices.clear()

    def restore_fit(self,arm):
        if self.fit is None:return None
        arm.matrix_world=self.matrix.copy()
        return self.fit.copy()

    def restore_residuals(self,garments):
        for obj,mod,values in self.offsets:
            changed=False
            for key,value in values.items():
                if tuple(mod[key])!=value:mod[key]=value;changed=True
            if changed:obj.update_tag()
        # Prevent a hidden filter from winding up while its output is held.
        # Visibility/last-seen times are deliberately NOT increased.
        garments.female_lock.values.update({i:v.copy() for i,v in self.values.items()})
