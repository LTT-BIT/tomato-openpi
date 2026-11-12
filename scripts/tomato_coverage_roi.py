"""Causal coverage-aware window; evidence and training validity remain distinct."""
import numpy as np

from scripts.tomato_state_guarded_roi import StableRoi as GuardedRoi
from scripts.tomato_stable_roi import HandTracker


class EvidenceHandTracker(HandTracker):
    """Strong current detections confirm identity without an unnecessary LK pass."""
    def update(self, gray, detections):
        height,width = gray.shape
        strong = []
        for detection in detections:
            box = np.asarray(detection['box'],float)
            score = detection['score']
            if (box.shape!=(4,) or not np.isfinite(box).all() or not np.isfinite(score)
                or not .85<=score<=1 or not (0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height)
                or min(box[2:]-box[:2])<15 or np.prod(box[2:]-box[:2])>width*height*.13):
                continue
            matched = self.box is None or np.linalg.norm((box[:2]+box[2:]-self.box[:2]-self.box[2:])/2)<.45*max(self.box[2:]-self.box[:2])
            if matched:
                strong.append((score,box))
        if strong:
            self.box = max(strong,key=lambda p:p[0])[1].copy()
            self.gray = gray.copy()
            self.missed = 0
            return self.box.copy(),'current_detector'
        return super().update(gray,detections)


class CoverageWindow:
    """Expand before an observed moving target reaches the edge, with bounded motion.

    Projection is onto the centers that contain the observed box plus margin.
    Velocity estimates use only the current and previous observed boxes. Abrupt
    target switches can still be invalid until the bounded window catches up.
    """
    def __init__(self):
        self.center = None
        self.side = None
        self.velocity = np.zeros(2)
        self.target_previous = None
        self.target_velocity = np.zeros(2)
        self.small_frames = 0

    def update(self, tight, width, height):
        minimum, maximum = .55*height, .95*min(width,height)
        if tight is not None:
            tight = np.asarray(tight, float)
            midpoint = (tight[:2]+tight[2:])/2
            if self.target_previous is not None:
                step = midpoint-self.target_previous
                # A role switch is not an object velocity observation.
                if np.linalg.norm(step)<.12*height:
                    self.target_velocity = .7*self.target_velocity+.3*step
                else:
                    self.target_velocity[:] = 0
            self.target_previous = midpoint.copy()
            desired_side = max(minimum,1.5*max(tight[2:]-tight[:2]))
            if self.center is not None:
                lead = np.clip(8*self.target_velocity,-.12*height,.12*height)
                future_lo = np.minimum(tight[:2],tight[:2]+lead)
                future_hi = np.maximum(tight[2:],tight[2:]+lead)
                required = 2*max(*(self.center-future_lo),*(future_hi-self.center))+12
                desired_side = max(desired_side,required)
            desired_side = float(np.clip(desired_side,minimum,maximum))
        else:
            midpoint = self.center.copy() if self.center is not None else np.array([.70*width,.65*height])
            desired_side = self.side if self.side is not None else minimum
            self.target_previous = None
            self.target_velocity[:] = 0
        if self.center is None:
            self.side = desired_side
            self.center = np.clip(midpoint,self.side/2,np.array([width,height])-self.side/2)
        else:
            self.small_frames = self.small_frames+1 if desired_side<.85*self.side else 0
            desired = desired_side if desired_side>self.side or self.small_frames>=15 else self.side
            self.side += float(np.clip(.16*(desired-self.side),-.008*self.side,.02*self.side))
            target = midpoint.copy()
            if tight is not None:
                # Any center in this interval covers the current evidence. Keep
                # the previous view when possible, but leave a 6px edge margin.
                margin = max(0.,min(6.,(self.side-max(tight[2:]-tight[:2]))/2))
                lo = tight[2:]-self.side/2+margin
                hi = tight[:2]+self.side/2-margin
                if np.all(lo<=hi):
                    target = np.clip(self.center+.25*(midpoint-self.center)+4*self.target_velocity,lo,hi)
            target = np.clip(target,self.side/2,np.array([width,height])-self.side/2)
            step = .3*(target-self.center)
            currently_covered = tight is not None and np.all(tight[:2]>=self.center-self.side/2) and np.all(tight[2:]<=self.center+self.side/2)
            if np.linalg.norm(target-self.center)<.006*self.side and (tight is None or currently_covered):
                step[:] = 0
            limit = .030*self.side
            step *= min(1.,limit/max(np.linalg.norm(step),1e-9))
            acceleration = step-self.velocity
            acceleration *= min(1.,.008*self.side/max(np.linalg.norm(acceleration),1e-9))
            self.velocity += acceleration
            self.velocity *= min(1.,limit/max(np.linalg.norm(self.velocity),1e-9))
            self.center = np.clip(self.center+self.velocity,self.side/2,np.array([width,height])-self.side/2)
        side = int(round(self.side))
        start = np.clip(np.round(self.center-side/2).astype(int),0,np.array([width-side,height-side]))
        box = np.r_[start,start+side]
        covered = tight is not None and bool(np.all(tight[:2]>=box[:2]) and np.all(tight[2:]<=box[2:]))
        return box,covered


class StableRoi(GuardedRoi):
    def __init__(self):
        super().__init__()
        self.window = CoverageWindow()
        self.hands = {role:EvidenceHandTracker() for role in (1,2)}

    def select(self, detections, state, width, height, rgb=None):
        return super().select(detections,state,width,height,rgb)
