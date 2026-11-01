"""Female-only motion response; the rendered stationary gate remains separate.

No future-frame prediction, confidence inflation, or smoothing of camera RGB.
Angles are radians, durations seconds. Capture time, not UI tick rate, drives
the follower so a slow computer does not add a second artificial turn delay.
"""
import math

def arc(value):
    return (value+math.pi)%(2*math.pi)-math.pi

class HeadingFollower:
    def __init__(self):self.reset()
    def reset(self):self.candidate=None
    def update(self,current,target,dt):
        if not all(math.isfinite(x) for x in (current,target,dt)) or dt<=0:
            return current
        dt=min(.25,max(.005,dt));error=arc(target-current)
        # Reject a lone impossible flip at high frame rates. A second similar
        # observation can confirm a rapid real turn; never latch permanently.
        limit=max(math.radians(35),dt*math.radians(900))
        if abs(error)>limit:
            confirmed=self.candidate is not None and abs(arc(target-self.candidate))<math.radians(25)
            self.candidate=target
            if not confirmed:return current
        else:self.candidate=None
        if abs(error)<math.radians(.7):return current
        # Fixed angle deadband, not the old 15--45 degree front suppression.
        # Slow noise is damped; a real turn uses the current observed heading.
        speed=abs(error)/dt
        tau=.018 if speed>math.radians(35) else .055
        step=error*(1-math.exp(-dt/tau))
        # Bound AFTER filtering. Bounding before alpha doubled the lag.
        step=max(-limit,min(limit,step))
        return current+step

def motion_confirmed(evidence,stamp):
    if not isinstance(evidence,dict) or evidence.get('valid') is not True or evidence.get('mode')!='moving':return False
    try:return abs(float(evidence.get('stamp',-1))-float(stamp))<.001
    except (TypeError,ValueError):return False

def loss_grace(female,best_stability):
    # At most a short one-packet grace for female turns. Past this, show the
    # live camera without an invented/stale garment, then reacquire normally.
    return .12 if female and best_stability else .65
