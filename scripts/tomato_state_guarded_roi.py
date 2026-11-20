"""Stable ROI with current measured finger-flexion evidence for left transport.

The state is an additional rejection cue, not proof that an object is held.
Current RGB object evidence remains required. The four raw state columns are the
left index/middle/ring/pinky MCP flexion positions, verified in source metadata.
"""
import numpy as np

from scripts.tomato_stable_roi import InteractionEvidence, StableRoi as VisualStableRoi, gap


LEFT_MCP_COLUMNS = (33,37,41,46)
LEFT_MCP_NAMES = tuple(f'left_{finger}_MCP_FE.position' for finger in ('index','middle','ring','pinky'))
GRASP_ENTER = .70
GRASP_EXIT = .58


class GuardedInteraction(InteractionEvidence):
    def __init__(self):
        super().__init__()
        self.closed = False
        self.close_count = 0
        self.open_count = 0
        self.curl = 0.0
        self.foreground_hits = 0
        self.foreground_previous = None
        self.image_size = (640,400)
        self.background = None
        self.arrival_count = 0

    def background_mask(self,regions):
        import cv2
        w,h=self.image_size
        occupancy=np.zeros((h,w),np.float32)
        for b in regions:
            x1,y1,x2,y2=np.rint(b).astype(int)
            occupancy[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]=1
        if self.background is None:
            self.background=occupancy.copy()
        static=cv2.dilate((self.background>.35).astype(np.uint8),np.ones((11,11),np.uint8))
        return occupancy,static

    @staticmethod
    def overlaps_background(box,mask):
        h,w=mask.shape
        x1,y1,x2,y2=np.rint(box).astype(int)
        patch=mask[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]
        return patch.size>0 and patch.mean()>.20

    def observe_state(self,state):
        state = np.asarray(state)
        if state.shape!=(116,) or not np.isfinite(state).all():
            raise ValueError('Finite measured116 state required')
        self.curl = float(state[list(LEFT_MCP_COLUMNS)].mean())
        self.close_count = self.close_count+1 if self.curl>=GRASP_ENTER else 0
        self.open_count = self.open_count+1 if self.curl<GRASP_EXIT else 0
        if self.close_count>=3:
            self.closed = True
        if self.open_count>=3:
            self.closed = False

    def choose(self,hands,regions):
        occupancy,background=self.background_mask(regions)
        if self.carried is not None and self.overlaps_background(self.carried,background):
            self.arrival_count+=1
        else:
            self.arrival_count=0
        if self.arrival_count>=15:
            self.carried=None
            self.previous_red=[]
            self.carry_hits=0
            self.carry_origin=None
            self.foreground_hits=0
            self.foreground_previous=None
        if self.carried is None:
            self.previous_red=[b for b in self.previous_red if not self.overlaps_background(b,background)]
        if not self.closed:
            self.carried = None
            self.previous_red = []
            self.carry_hits = 0
            self.carry_origin = None
            self.foreground_hits = 0
            self.foreground_previous = None
            if self.roles==(2,):
                self.roles = (1,)
                self.dwell = 0
            if self.pending==(2,):
                self.pending = None
                self.pending_count = 0
        elif 2 in hands:
            left=hands[2]
            candidates=[]
            for b in regions:
                center=(b[:2]+b[2:])/2
                stationary=any(age>=4 and np.linalg.norm(center-anchor)<2.5 for _,age,anchor in self.stationary_red)
                if (not stationary and not self.overlaps_background(b,background) and np.prod(b[2:]-b[:2])>=20 and gap(b,left)<.12*max(left[2:]-left[:2])
                    and (1 not in hands or gap(b,hands[1])>.20*max(hands[1][2:]-hands[1][:2]))):
                    candidates.append(b)
            if candidates:
                current=min(candidates,key=lambda b:gap(b,left))
                center=(current[:2]+current[2:])/2
                consistent=self.foreground_previous is not None and np.linalg.norm(center-self.foreground_previous)<.5*max(left[2:]-left[:2])
                self.foreground_hits=self.foreground_hits+1 if consistent else 1
                self.foreground_previous=center.copy()
                if self.foreground_hits>=3:
                    self.carried=current.copy()
                    self.carry_lost=0
            else:
                self.foreground_hits=0
                self.foreground_previous=None
        roles = super().choose(hands,regions)
        # Do not label an empty left-hand dwell as confident transport after release.
        if roles==(2,) and self.carried is None:
            self.roles = (1,)
            self.pending = None
            self.pending_count = 0
            self.dwell = 0
        if self.carried is None:
            self.previous_red=[b for b in self.previous_red if not self.overlaps_background(b,background)]
        updated=.98*self.background+.02*occupancy
        # A held object becoming stationary in the air must not turn into background.
        if self.closed and self.carried is not None and 2 in hands:
            w,h=self.image_size
            x1,y1,x2,y2=np.rint(hands[2]+np.array([-10,-10,10,10])).astype(int)
            area=np.s_[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]
            updated[area]=self.background[area]
        self.background=updated
        return self.roles


class StableRoi(VisualStableRoi):
    def __init__(self):
        super().__init__()
        self.interaction = GuardedInteraction()

    def select(self,detections,state,width,height,rgb=None):
        self.interaction.image_size=(width,height)
        self.interaction.observe_state(state)
        result = super().select(detections,state,width,height,rgb)
        result['left_hand_mcp_fe_mean'] = self.interaction.curl
        result['left_grasp_state_confirmed'] = self.interaction.closed
        result['support_arrival_frames'] = self.interaction.arrival_count
        return result
