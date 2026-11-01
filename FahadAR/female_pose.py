"""Evaluate the unconstrained female rig without evaluating all garment meshes.

Blender's own Bone conversion handles rest matrices and inherit-scale rules.
Unsupported evaluated dependencies explicitly fall back to the existing path.
The final scene evaluation still happens before RGB/matte are committed.
"""
import numpy as np

def control_heads(arm,base,names):
    animation=arm.animation_data
    if animation and (animation.drivers or animation.action or animation.nla_tracks):return None
    if any(b.constraints or getattr(b.bone,'use_bbone_custom_handles',False) for b in arm.pose.bones):return None
    matrices={}
    def pose_matrix(bone):
        cached=matrices.get(bone.name)
        if cached is not None:return cached
        kwargs={} if bone.parent is None else dict(
            parent_matrix=pose_matrix(bone.parent),parent_matrix_local=bone.parent.bone.matrix_local)
        result=bone.bone.convert_local_to_pose(bone.matrix_basis,bone.bone.matrix_local,**kwargs)
        matrices[bone.name]=result
        return result
    try:
        return {i:np.array(base@pose_matrix(arm.pose.bones['mixamorig:'+name]).translation) for i,name in names.items()}
    except (KeyError,AttributeError,TypeError):return None
