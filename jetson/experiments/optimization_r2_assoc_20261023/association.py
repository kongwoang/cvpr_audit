"""GT-free identity checks for the one-object historical recovery budget."""
import time
import cv2
import numpy as np
from rap.mono import box_iou

def margin_match(box,coarse,det,min_iou=.5,min_margin=.2):
    scores=box_iou(np.asarray(box).reshape(1,4),np.asarray(det['xyxy']).reshape(-1,4))[0]
    ids=np.flatnonzero(np.asarray(det['coarse'])==coarse)
    if not len(ids):return -1
    ids=ids[np.argsort(-scores[ids],kind='stable')];best=int(ids[0])
    second=float(scores[ids[1]]) if len(ids)>1 else 0.
    return best if scores[best]>=min_iou and scores[best]-second>=min_margin else -1

def backward_box(current,previous,box):
    """Local bidirectional LK + robust similarity; reject weak/inconsistent flow.

    No detection or GT enters the optical-flow fit. A box prediction locates a
    query only; it must never become an observed detection/C1 history entry.
    """
    start=time.perf_counter();box=np.asarray(box,float);H,W=current.shape[:2]
    assert previous.shape[:2]==(H,W)
    centre=(box[:2]+box[2:])/2;size=box[2:]-box[:2]
    info={'success':False,'reason':'small_box','points':0,'inliers':0}
    def done(result):
        info['cpu_ms']=(time.perf_counter()-start)*1000
        return result,info
    if min(size)<12:return done(None)
    x1,y1=np.maximum(np.floor(centre-2*size),0).astype(int)
    x2,y2=np.minimum(np.ceil(centre+2*size),[W,H]).astype(int)
    scale=min(1.,384/max(x2-x1,y2-y1))
    def gray(image):
        a=cv2.cvtColor(image[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY)
        return cv2.resize(a,None,fx=scale,fy=scale) if scale<1 else a
    a,b=gray(current),gray(previous);info['roi_pixels']=int(a.size)
    mask=np.zeros_like(a);bb=(box-np.array([x1,y1,x1,y1]))*scale
    left,top=np.ceil(bb[:2]+.05*(bb[2:]-bb[:2])).astype(int)
    right,bottom=np.floor(bb[2:]-.05*(bb[2:]-bb[:2])).astype(int)
    mask[max(0,top):min(bottom,a.shape[0]),max(0,left):min(right,a.shape[1])]=255
    p=cv2.goodFeaturesToTrack(a,maxCorners=40,qualityLevel=.01,minDistance=3,mask=mask,blockSize=3)
    info['reason']='too_few_corners'
    if p is None or len(p)<6:return done(None)
    args=dict(winSize=(15,15),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    q,ok,_=cv2.calcOpticalFlowPyrLK(a,b,p,None,**args)
    back,ok_back,_=cv2.calcOpticalFlowPyrLK(b,a,q,None,**args)
    fb=np.linalg.norm((back-p).reshape(-1,2),axis=1)
    good=ok.ravel().astype(bool)&ok_back.ravel().astype(bool)&(fb<=1.)
    info.update(points=int(len(p)),fb_consistent=int(good.sum()),reason='inconsistent_flow')
    if good.sum()<6:return done(None)
    pp,qq=p.reshape(-1,2)[good],q.reshape(-1,2)[good]
    cv2.setRNGSeed(0)
    M,inlier=cv2.estimateAffinePartial2D(pp,qq,method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=500,confidence=.99,refineIters=10)
    if M is None:return done(None)
    nin=int(inlier.sum());zoom=float(np.hypot(M[0,0],M[0,1]))
    info.update(inliers=nin,scale=zoom,reason='weak_or_extreme_fit')
    if nin<6 or nin/len(pp)<.6 or not .75<=zoom<=1.33:return done(None)
    corners=np.array([[bb[0],bb[1],1],[bb[2],bb[1],1],[bb[2],bb[3],1],[bb[0],bb[3],1]])
    pred=corners@M.T/scale+np.array([x1,y1]);xy=np.r_[pred.min(0),pred.max(0)]
    xy[:2]=np.maximum(xy[:2],0);xy[2:]=np.minimum(xy[2:],[W,H])
    if np.any(xy[2:]-xy[:2]<8):return done(None)
    info.update(success=True,reason='accepted',displacement=((xy[:2]+xy[2:]-box[:2]-box[2:])/2).tolist())
    return done(xy)

def shifted_canvas(key,current_box,past_box,W,H):
    """Translate the same-size historical query; preserve network canvas cost."""
    x1,y1,x2,y2,nh,nw=key
    width=min(x2,W)-x1;height=min(y2,H)-y1
    delta=(np.asarray(past_box)[:2]+np.asarray(past_box)[2:]-np.asarray(current_box)[:2]-np.asarray(current_box)[2:])/2
    left=int(np.clip(round(x1+delta[0]),0,W-width));top=int(np.clip(round(y1+delta[1]),0,H-height))
    return left,top,left+width,top+height,nh,nw
