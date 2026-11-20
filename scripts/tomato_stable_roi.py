"""Causal hand evidence, persistent interaction selection and bounded crop motion.

No phase labels or future images are inputs. A continuous display is deliberately
separate from trustworthy ROI validity: current pixels remain visible on loss,
while the training mask is false and training ROI coordinates are null.
"""
import numpy as np


def union(boxes):
    boxes = np.asarray(boxes, float)
    return np.r_[boxes[:, :2].min(0), boxes[:, 2:].max(0)]


def gap(a, b):
    return float(np.linalg.norm(np.maximum(0, np.maximum(a[:2], b[:2])-np.minimum(a[2:], b[2:]))))


class HandTracker:
    """A lower continuation threshold requires spatial agreement with past identity.

    Optical-flow bridging must be verified against the CURRENT image and expires
    after five frames without detector confirmation. Never reuses stale pixels.
    """
    def __init__(self):
        self.gray = None
        self.box = None
        self.missed = 0

    def flow_box(self, gray):
        import cv2
        if self.gray is None or self.box is None or self.missed >= 5:
            return None
        x1,y1,x2,y2 = self.box
        mask = np.zeros_like(gray)
        mask[int(y1):int(y2),int(x1):int(x2)] = 255
        points = cv2.goodFeaturesToTrack(self.gray, 60, .02, 4, mask=mask)
        if points is None or len(points) < 8:
            return None
        current, ok, err = cv2.calcOpticalFlowPyrLK(self.gray, gray, points, None, winSize=(21,21), maxLevel=3)
        if current is None:
            return None
        previous, back_ok, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, current, None, winSize=(21,21), maxLevel=3)
        if previous is None:
            return None
        good = (ok.ravel()>0)&(back_ok.ravel()>0)&(np.linalg.norm(previous-points,axis=2).ravel()<1.0)&(err.ravel()<20)
        if good.sum() < 8 or good.mean() < .65:
            return None
        matrix, inliers = cv2.estimateAffinePartial2D(points[good],current[good],method=cv2.RANSAC,ransacReprojThreshold=1.5)
        if matrix is None or inliers.mean() < .75:
            return None
        scale = np.linalg.norm(matrix[:,0])
        if not .97 <= scale <= 1.03:
            return None
        corners = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],float)
        warped = corners@matrix[:,:2].T+matrix[:,2]
        box = np.r_[warped.min(0),warped.max(0)]
        h,w = gray.shape
        if not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):
            return None
        if np.prod(box[2:]-box[:2])>w*h*.13:
            return None
        if np.linalg.norm((box[:2]+box[2:]-self.box[:2]-self.box[2:])/2) > .35*max(x2-x1,y2-y1):
            return None
        return box

    def update(self, gray, detections):
        h,w = gray.shape
        flow = self.flow_box(gray)
        candidates = []
        for item in detections:
            b = np.asarray(item['box'],float)
            score = item['score']
            if (b.shape!=(4,) or not np.isfinite(b).all() or not np.isfinite(score)
                or not 0<=score<=1 or not (0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h)
                or min(b[2:]-b[:2])<15 or np.prod(b[2:]-b[:2])>w*h*.13):
                continue
            reference = flow if flow is not None else self.box
            matched = reference is not None and np.linalg.norm((b[:2]+b[2:]-reference[:2]-reference[2:])/2)<.45*max(reference[2:]-reference[:2])
            if flow is not None and not matched:
                continue
            if score >= (.55 if matched else .75):
                candidates.append((score+.12*matched,b))
        if candidates:
            self.box = max(candidates,key=lambda item:item[0])[1].copy()
            self.missed = 0
            source = 'current_detector'
        elif flow is not None:
            self.box = flow
            self.missed += 1
            source = 'verified_current_flow'
        else:
            self.box = None
            self.missed += 1
            source = 'lost'
        self.gray = gray.copy()
        return None if self.box is None else self.box.copy(), source


def red_regions(rgb):
    import cv2
    hsv = cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    mask = (((hsv[:,:,0]<13)|(hsv[:,:,0]>169))&(hsv[:,:,1]>100)&(hsv[:,:,2]>65)).astype(np.uint8)
    _,_,stats,_ = cv2.connectedComponentsWithStats(mask,8)
    return [np.array([x,y,x+w,y+h],float) for x,y,w,h,area in stats[1:]
            if 12<=area<=2500 and min(w,h)>=3]


class InteractionEvidence:
    """Left transport requires red evidence moving WITH the hand, not arm speed alone."""
    def __init__(self):
        self.previous_left = None
        self.previous_red = []
        self.stationary_red = []
        self.carried = None
        self.carry_hits = 0
        self.carry_origin = None
        self.carry_lost = 0
        self.roles = (1,)
        self.pending = None
        self.pending_count = 0
        self.dwell = 30

    def choose(self, hands, regions):
        left = hands.get(2)
        left_near = [] if left is None else [b for b in regions if gap(b,left)<.2*max(left[2:]-left[:2])]
        stationary = []
        for current in regions:
            center = (current[:2]+current[2:])/2
            matching = [(age,anchor) for old,age,anchor in self.stationary_red
                        if np.linalg.norm(center-anchor)<2.5]
            age,anchor = max(matching,key=lambda item:item[0]) if matching else (0,center.copy())
            stationary.append((current.copy(),min(age+1,30),anchor.copy()))
        self.stationary_red = stationary
        attached = None
        if left is not None and self.previous_left is not None:
            motion = (left[:2]+left[2:]-self.previous_left[:2]-self.previous_left[2:])/2
            candidates = self.previous_red if self.carried is None else [self.carried]
            for old in candidates:
                old_center = (old[:2]+old[2:])/2
                for current in left_near:
                    shift = (current[:2]+current[2:])/2-old_center
                    # Static fruit in a basket cannot initiate left transport.
                    if self.carried is None and (np.linalg.norm(motion)<1.0 or np.linalg.norm(shift)<.7):
                        continue
                    if self.carried is None and any(age>=5 and np.array_equal(current,b) for b,age,_ in stationary):
                        continue
                    if not .6<=np.prod(current[2:]-current[:2])/np.prod(old[2:]-old[:2])<=1.8:
                        continue
                    if np.linalg.norm(shift-motion)<max(3.0,.08*max(left[2:]-left[:2])):
                        attached = current
                        break
                if attached is not None:
                    break
        if attached is not None:
            if self.carry_hits==0:
                self.carry_origin = (attached[:2]+attached[2:])/2
            self.carry_hits += 1
            self.carry_lost = 0
            displacement = np.linalg.norm((attached[:2]+attached[2:])/2-self.carry_origin)
            if self.carried is not None or (self.carry_hits>=3 and displacement>=5):
                self.carried = attached.copy()
        else:
            self.carry_hits = 0
            self.carry_origin = None
            self.carry_lost += 1
            if self.carry_lost>=6:
                self.carried = None
        close = False
        if 1 in hands and 2 in hands:
            a,b = hands[1],hands[2]
            threshold = .75 if self.roles==(1,2) else .40
            close = gap(a,b)<threshold*max(*(a[2:]-a[:2]),*(b[2:]-b[:2]))
        proposal = (1,2) if close else ((2,) if self.carried is not None else (1,))
        self.dwell += 1
        if proposal == self.roles:
            self.pending = None
            self.pending_count = 0
        else:
            self.pending_count = self.pending_count+1 if proposal==self.pending else 1
            self.pending = proposal
            if self.pending_count>=6 and self.dwell>=12:
                self.roles = proposal
                self.pending = None
                self.pending_count = 0
                self.dwell = 0
        self.previous_left = None if left is None else left.copy()
        self.previous_red = [b.copy() for b in left_near]
        return self.roles


class SmoothWindow:
    """Causal bounded camera motion. Coverage failure remains visible as false mask."""
    def __init__(self):
        self.center = None
        self.side = None
        self.velocity = np.zeros(2)
        self.small_frames = 0

    def update(self, tight, width, height):
        minimum = .55*height
        maximum = .95*min(width,height)
        if tight is not None:
            target_center = (tight[:2]+tight[2:])/2
            target_side = float(np.clip(1.5*max(tight[2:]-tight[:2]),minimum,maximum))
        elif self.center is None:
            target_center = np.array([.70*width,.65*height])
            target_side = minimum
        else:
            target_center, target_side = self.center.copy(), self.side
        if self.center is None:
            self.side = target_side
            self.center = np.clip(target_center,self.side/2,np.array([width,height])-self.side/2)
        else:
            if target_side<.88*self.side:
                self.small_frames += 1
            else:
                self.small_frames = 0
            desired = target_side if target_side>1.08*self.side or self.small_frames>=15 else self.side
            self.side += float(np.clip(.08*(desired-self.side),-.004*self.side,.010*self.side))
            target_center = np.clip(target_center,self.side/2,np.array([width,height])-self.side/2)
            velocity = .16*(target_center-self.center)
            if np.linalg.norm(target_center-self.center)<.012*self.side:
                velocity = np.zeros(2)
            norm = np.linalg.norm(velocity)
            limit = .020*self.side
            if norm>limit:
                velocity *= limit/norm
            acceleration = velocity-self.velocity
            norm = np.linalg.norm(acceleration)
            if norm>.004*self.side:
                acceleration *= (.004*self.side)/norm
            self.velocity += acceleration
            # Clamp after acceleration too: side may have shrunk since the last step.
            norm = np.linalg.norm(self.velocity)
            if norm>limit:
                self.velocity *= limit/norm
            self.center += self.velocity
            self.center = np.clip(self.center,self.side/2,np.array([width,height])-self.side/2)
        # Quantize only the final sampling window, retaining continuous internal state.
        # Exact integer squares are compatible with native_crop and the data contract.
        side = int(round(self.side))
        start = np.clip(np.round(self.center-side/2).astype(int),0,np.array([width-side,height-side]))
        box = np.r_[start,start+side]
        covered = tight is not None and bool(np.all(tight[:2]>=box[:2]) and np.all(tight[2:]<=box[2:]))
        return box,covered


class StableRoi:
    def __init__(self):
        self.hands = {role:HandTracker() for role in (1,2)}
        self.interaction = InteractionEvidence()
        self.window = SmoothWindow()

    def select(self,detections,state,width,height,rgb=None):
        import cv2
        state = np.asarray(state)
        if state.shape!=(116,) or not np.isfinite(state).all():
            raise ValueError('Finite current measured116 state required')
        if rgb is None or rgb.shape!=(height,width,3):
            raise ValueError('Current native RGB required')
        gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        hands = {}
        evidence = {}
        for role,tracker in self.hands.items():
            box,source = tracker.update(gray,[d for d in detections if d['label']==role])
            evidence[str(role)] = source
            if box is not None:
                hands[role] = box
        regions = red_regions(rgb)
        roles = self.interaction.choose(hands,regions)
        tight = None
        if all(role in hands for role in roles):
            boxes = [hands[role] for role in roles]
            # Limit color association to the immediate hand; never chain through a basket.
            boxes += [b for b in regions if any(gap(b,hands[role])<.12*max(hands[role][2:]-hands[role][:2]) for role in roles)]
            tight = union(boxes)
        display, covered = self.window.update(tight,width,height)
        valid = tight is not None and covered
        return dict(valid=valid,roles=list(roles),interaction_box=tight.tolist() if valid else None,
                    crop_box=display.tolist() if valid else None,display_crop_box=display.tolist(),
                    observed_interaction_box=None if tight is None else tight.tolist(),
                    reason=None if valid else ('window_catching_up' if tight is not None else 'required_hand_lost'),
                    hand_evidence=evidence,carry_evidence=self.interaction.carried is not None,
                    validity_scope='current visual/flow evidence and full geometric coverage; semantics still require QA')
