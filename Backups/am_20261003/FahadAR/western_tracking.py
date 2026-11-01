"""Bounded lower-visibility continuity for the three reference Western outfits."""
import numpy as np
import math

def follow_heading(current,target,dt):
    """Shortest-arc heading, with speed limiting AFTER temporal smoothing.

    Limiting first and then multiplying by alpha made a 180 deg/s limit
    behave like ~51 deg/s at 30 Hz, trailing ordinary body turns badly.
    """
    dt=max(.001,min(.25,float(dt)))
    error=(target-current+math.pi)%(2*math.pi)-math.pi
    if abs(error)<math.radians(.7):return current
    alpha=1-math.exp(-dt/(.035 if abs(error)>math.radians(5) else .09))
    return current+float(np.clip(error*alpha,-dt*math.radians(240),dt*math.radians(240)))

class LowerVisibility:
    def __init__(self):self.reset()
    def reset(self):self.last_good=None;self.last_stamp=None
    def update(self,pose,stamp,calibrated):
        stamp=float(stamp)
        if not calibrated or not np.isfinite(stamp):
            self.reset();return False
        if self.last_stamp is not None and stamp<self.last_stamp:self.reset()
        self.last_stamp=stamp
        p=np.asarray(pose,float)
        if p.shape!=(33,4) or not np.isfinite(p).all():return False
        valid=(p[:,3]>=.5)&(p[:,0]>.005)&(p[:,0]<.995)&(p[:,1]>.005)&(p[:,1]<.995)
        if not any(valid[a] and valid[b] for a,b in ((11,23),(12,24))):
            self.last_good=None;return False
        if any(all(valid[i] for i in side) for side in ((23,25,27),(24,26,28))):
            self.last_good=stamp;return True
        # Not a new confidence measurement. Only hide/show hysteresis during
        # the same short dropout covered by residual decay; shoes stay strict.
        return self.last_good is not None and 0<=stamp-self.last_good<=.45
