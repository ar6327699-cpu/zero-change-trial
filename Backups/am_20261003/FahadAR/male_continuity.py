"""Independent male state using the proven, unchanged continuity primitives.

Female modules are read-only dependencies; no global state is patched. Each
camera/session constructs its own trackers. The old male landmark gate remains
the fallback when image texture or visibility cannot support a visual hold.
"""
from female_continuity import HeadingFollower, motion_confirmed
from female_precision import StillnessGate as ImageStillnessGate
from male_tracking import StillnessGate as LandmarkStillnessGate

class StillnessGate(ImageStillnessGate):
    def reset(self):
        super().reset()
        self.fallback=LandmarkStillnessGate()

def ImageMotion():
    # OpenCV belongs to the camera worker's interpreter, not Blender's.
    from female_motion import ImageMotion as Tracker
    return Tracker()

def control_heads(arm,base,names):
    # Pure bone conversion; importing does not change female objects or state.
    from female_pose import control_heads as convert
    return convert(arm,base,names)

LOSS_GRACE=.12
