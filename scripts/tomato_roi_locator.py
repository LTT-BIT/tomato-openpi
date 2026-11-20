"""Experimental current-RGB locator and causal measured-state role selection.

Algorithmic validity is not independent semantic quality approval. No stage
labels, future frames, or action commands are accepted by the selector.
"""
from pathlib import Path
import json
import numpy as np
from scripts.tomato_interaction_preview import square_box


def make_locator_model():
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model=fasterrcnn_mobilenet_v3_large_320_fpn(weights=None,weights_backbone=None)
    model.roi_heads.box_predictor=FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features,4)
    return model


class RgbLocator:
    def __init__(self,root,device='cuda'):
        import torch
        root=Path(root);receipt=json.loads((root/'receipt.json').read_text())
        if receipt.get('status')!='experimental_training_finished_no_independent_quality_acceptance':
            raise ValueError('Completed isolated locator receipt required')
        payload=torch.load(root/'model.pt',map_location='cpu',weights_only=True)
        if payload['classes']!={1:'right_hand',2:'left_hand',3:'held_object'}:
            raise ValueError('Unexpected locator class contract')
        self.model=make_locator_model();self.model.load_state_dict(payload['state_dict'])
        self.model=self.model.to(device).eval();self.device=device

    def predict(self,images):
        import torch
        inputs=[torch.from_numpy(np.array(im,copy=True)).permute(2,0,1).float().div(255).to(self.device) for im in images]
        with torch.inference_mode():outputs=self.model(inputs)
        return [[dict(label=int(label),score=float(score),box=box.tolist())
                 for box,label,score in zip(p['boxes'].cpu(),p['labels'].cpu(),p['scores'].cpu(),strict=True) if score>=.5]
                for p in outputs]


class CausalRoleSelector:
    def __init__(self,state_scale):
        self.scale=np.maximum(np.asarray(state_scale),1e-4)
        if self.scale.shape!=(58,) or not np.isfinite(self.scale).all():raise ValueError('Finite58 state scale required')
        self.previous=None;self.speed=np.zeros(2);self.role=1

    def select(self,detections,state,width,height,rgb=None):
        state=np.asarray(state)
        if state.shape!=(116,) or not np.isfinite(state).all():raise ValueError('Finite measured116 state required')
        q=np.r_[state[:7],state[28:50],state[14:21],state[72:94]]
        if self.previous is not None:
            delta=np.minimum(np.abs(q-self.previous)/self.scale,.2)
            motion=np.array([np.sqrt(np.mean(delta[:7]**2))+.2*np.sqrt(np.mean(delta[7:29]**2)),
                             np.sqrt(np.mean(delta[29:36]**2))+.2*np.sqrt(np.mean(delta[36:]**2))])
            self.speed=.4*motion+.6*self.speed
            if max(self.speed)>.0005:
                proposal=2 if self.speed[0]>self.speed[1] else 1
                current_speed=self.speed[0 if self.role==2 else 1]
                if max(self.speed)>1.4*current_speed:self.role=proposal
        self.previous=q.copy()
        best={}
        for d in detections:
            label=d['label'];score=d['score'];b=np.asarray(d['box'],float)
            if label not in (1,2,3) or not np.isfinite(score) or not .0<=score<=1 or b.shape!=(4,) or not np.isfinite(b).all():continue
            if score<(.75 if label in (1,2) else .6):continue
            if not (0<=b[0]<b[2]<=width and 0<=b[1]<b[3]<=height):continue
            area=np.prod(b[2:]-b[:2])
            if label in (1,2) and (area>width*height*.13 or min(b[2:]-b[:2])<15):continue
            if label==3 and area>width*height*.08:continue
            if label not in best or score>best[label]['score']:best[label]=d
        roles=[self.role]
        if 1 in best and 2 in best:
            a=np.array(best[1]['box']);b=np.array(best[2]['box'])
            gap=np.linalg.norm(np.maximum(0,np.maximum(a[:2],b[:2])-np.minimum(a[2:],b[2:])))
            if gap<.5*max(*(a[2:]-a[:2]),*(b[2:]-b[:2])):roles=[1,2]
        if any(role not in best for role in roles):
            return dict(valid=False,crop_box=None,interaction_box=None,roles=roles,reason='required_hand_not_confidently_located')
        boxes=np.array([best[role]['box'] for role in roles])
        tight=np.r_[boxes[:,:2].min(0),boxes[:,2:].max(0)]
        if 3 in best:
            obj=np.array(best[3]['box'])
            gap=np.linalg.norm(np.maximum(0,np.maximum(tight[:2],obj[:2])-np.minimum(tight[2:],obj[2:])))
            if gap<.35*max(tight[2:]-tight[:2]):tight=np.r_[np.minimum(tight[:2],obj[:2]),np.maximum(tight[2:],obj[2:])]
        if rgb is not None:
            for hand in boxes:
                for red in near_red_regions(rgb,hand):
                    tight=np.r_[np.minimum(tight[:2],red[:2]),np.maximum(tight[2:],red[2:])]
        return dict(valid=True,crop_box=list(square_box(tight,width,height)),interaction_box=tight.tolist(),roles=roles,
                    reason=None,validity_scope='detector/motion heuristic, semantic quality unapproved')


def near_red_regions(rgb,hand_box):
    """Current RGB evidence near a hand; red color is a heuristic, not object identity proof."""
    import cv2
    image=np.asarray(rgb)
    hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    mask=(((hsv[:,:,0]<13)|(hsv[:,:,0]>169))&(hsv[:,:,1]>90)&(hsv[:,:,2]>60)).astype(np.uint8)
    _,_,stats,_=cv2.connectedComponentsWithStats(mask,8)
    candidates=[np.array([x,y,x+w,y+h],float) for x,y,w,h,area in stats[1:] if 8<=area<=3000 and min(w,h)>=3]
    selected=[];active=np.asarray(hand_box,float).copy();first=True
    while candidates:
        changed=False;remaining=[]
        margin=.4*max(active[2:]-active[:2]) if first else 12
        for b in candidates:
            gap=np.linalg.norm(np.maximum(0,np.maximum(active[:2],b[:2])-np.minimum(active[2:],b[2:])))
            merged=np.r_[np.minimum(active[:2],b[:2]),np.maximum(active[2:],b[2:])]
            if gap<=margin and np.prod(merged[2:]-merged[:2])<image.shape[0]*image.shape[1]*.25:
                selected.append(b);active=merged;changed=True
            else:remaining.append(b)
        if not changed:break
        candidates=remaining;first=False
    return selected
