"""Offline temporal ROI repair. Future-frame annotation support is explicit.

Only coordinates use neighboring frames. Every exported image must be cut from
its own native frame; these annotations are not a causal deployment locator.
"""
import numpy as np


def union(boxes):
    a=np.asarray(boxes,float)
    return np.r_[a[:,:2].min(0),a[:,2:].max(0)]


def flow_step(previous,current,box):
    """Validate motion against actual adjacent images, including reverse flow."""
    import cv2
    h,w=previous.shape
    x1,y1,x2,y2=np.rint(box).astype(int)
    mask=np.zeros_like(previous)
    mask[max(0,y1+1):min(h,y2-1),max(0,x1+1):min(w,x2-1)]=255
    points=cv2.goodFeaturesToTrack(previous,100,.01,3,mask=mask)
    if points is None or len(points)<6:return None
    q,ok,error=cv2.calcOpticalFlowPyrLK(previous,current,points,None,winSize=(21,21),maxLevel=3)
    if q is None:return None
    back,bok,_=cv2.calcOpticalFlowPyrLK(current,previous,q,None,winSize=(21,21),maxLevel=3)
    if back is None:return None
    good=(ok.ravel()>0)&(bok.ravel()>0)&(np.linalg.norm(back-points,axis=2).ravel()<1)&(error.ravel()<25)
    if good.sum()<6 or good.mean()<.60:return None
    matrix,inliers=cv2.estimateAffinePartial2D(points[good],q[good],method=cv2.RANSAC,ransacReprojThreshold=1.5)
    if matrix is None or inliers.mean()<.70:return None
    scale=np.linalg.norm(matrix[:,0])
    if not .97<=scale<=1.03:return None
    # Follow translation; endpoint boxes bound finger pose/scale changes.
    shift=np.median((q-points)[good],axis=0).ravel()
    if np.linalg.norm(shift)>.25*max(x2-x1,y2-y1):return None
    result=np.asarray(box,float)+np.r_[shift,shift]
    if not (0<=result[0]<result[2]<=w and 0<=result[1]<result[3]<=h):return None
    return result


def track(gray,seed_frame,seed_box,frames):
    boxes={seed_frame:np.asarray(seed_box,float)}
    previous=seed_frame;box=np.asarray(seed_box,float)
    for frame in frames:
        candidate=flow_step(gray[previous],gray[frame],box)
        if candidate is None:candidate=template_step(gray[previous],gray[frame],box)
        if candidate is None:break
        boxes[frame]=candidate;previous=frame;box=candidate
    return boxes


def template_step(previous,current,box):
    """A high-correlation local patch can bridge low-texture LK failures."""
    import cv2
    h,w=previous.shape
    x1,y1,x2,y2=np.rint(box).astype(int)
    x1,y1,x2,y2=max(0,x1),max(0,y1),min(w,x2),min(h,y2)
    patch=previous[y1:y2,x1:x2]
    if min(patch.shape,default=0)<15 or patch.std()<10:return None
    margin=max(8,int(.20*max(x2-x1,y2-y1)))
    a,b=max(0,x1-margin),max(0,y1-margin)
    search=current[b:min(h,y2+margin),a:min(w,x2+margin)]
    correlation=cv2.matchTemplate(search,patch,cv2.TM_CCOEFF_NORMED)
    _,score,_,position=cv2.minMaxLoc(correlation)
    if score<.85:return None
    shift=np.array([a+position[0]-x1,b+position[1]-y1],float)
    return np.asarray(box,float)+np.r_[shift,shift]


def plan_windows(tight,width,height):
    """Look ahead offline to pan/zoom before a target transition; contain every box."""
    from scipy.ndimage import gaussian_filter1d
    tight=np.asarray(tight,float)
    if tight.ndim!=2 or tight.shape[1]!=4 or not np.isfinite(tight).all():
        raise ValueError('Finite complete interaction boxes required')
    if np.any(tight[:,2:]<=tight[:,:2]):raise ValueError('Empty interaction box')
    minimum,maximum=220.,min(width,height)*.95
    if np.max(tight[:,2:]-tight[:,:2])>maximum-2:
        raise ValueError('Interaction is too large for local crop limit')
    desired=(tight[:,:2]+tight[:,2:])/2
    center=gaussian_filter1d(desired,15,axis=0,mode='nearest')
    # Feasible centers at the maximum local scale, with a pixel for rounding.
    lo=np.maximum(maximum/2,tight[:,2:]-maximum/2+1)
    hi=np.minimum(np.array([width,height])-maximum/2,tight[:,:2]+maximum/2-1)
    # Propagate future feasible intervals before projecting the center. Merely
    # clipping a smoothed center can jump when a two-hand constraint disappears.
    speed=3.
    for i in range(1,len(lo)):
        lo[i]=np.maximum(lo[i],lo[i-1]-speed)
        hi[i]=np.minimum(hi[i],hi[i-1]+speed)
    for i in range(len(lo)-2,-1,-1):
        lo[i]=np.maximum(lo[i],lo[i+1]-speed)
        hi[i]=np.minimum(hi[i],hi[i+1]+speed)
    if np.any(lo>hi):raise ValueError('No bounded local trajectory contains all annotations')
    forward=np.clip(center,lo,hi)
    for i in range(1,len(center)):
        forward[i]=np.clip(center[i],np.maximum(lo[i],forward[i-1]-speed),np.minimum(hi[i],forward[i-1]+speed))
    backward=np.clip(center,lo,hi)
    for i in range(len(center)-2,-1,-1):
        backward[i]=np.clip(center[i],np.maximum(lo[i],backward[i+1]-speed),np.minimum(hi[i],backward[i+1]+speed))
    center=(forward+backward)/2
    required=np.maximum(minimum,2*np.max(np.maximum(center-tight[:,:2],tight[:,2:]-center),axis=1)+2)
    preferred=np.maximum(required,np.minimum(maximum,1.5*np.max(tight[:,2:]-tight[:,:2],axis=1)))
    # A symmetric envelope expands before a transition and contracts afterwards.
    side=preferred.copy()
    decay=np.exp(-.008)
    for i in range(1,len(side)):side[i]=max(side[i],side[i-1]*decay)
    for i in range(len(side)-2,-1,-1):side[i]=max(side[i],side[i+1]*decay)
    side=np.minimum(maximum,np.ceil(side)).astype(int)
    start=np.rint(center-side[:,None]/2).astype(int)
    start=np.maximum(0,np.minimum(start,np.array([width,height])-side[:,None]))
    boxes=np.c_[start,start+side[:,None]]
    contained=np.all(boxes[:,:2]<=tight[:,:2],axis=1)&np.all(boxes[:,2:]>=tight[:,2:],axis=1)
    centers=(boxes[:,:2]+boxes[:,2:])/2
    steps=np.linalg.norm(np.diff(centers,axis=0),axis=1)
    ratios=np.abs(np.diff(side))/side[:-1]
    return boxes,dict(all_boxes_contained=bool(contained.all()),uncovered_frames=np.flatnonzero(~contained).tolist(),
                      max_center_step_px=float(steps.max(initial=0)),center_steps_over_3pct=int((steps>.03*side[:-1]).sum()),
                      max_scale_step_fraction=float(ratios.max(initial=0)),scale_steps_over_2pct=int((ratios>.02).sum()))
