import cv2
import numpy as np
from collections import deque
class OptFlow():
    def __init__(self,flow_type='sparse',qlen=125):
        # params for ShiTomasi corner detection
        self.feature_params = dict( maxCorners = 100,
                               qualityLevel = 0.3,
                               minDistance = 7,
                               blockSize = 7 )

        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (15,15),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        
        self.background = None
        self.max_frames = 1000
        self.thresh = 60
        self.assign_value = 255
        self.alpha = 0.1
        
        # Create some random colors
        self.color      = np.random.randint(0,255,(100,3))
        self.prev_frame = []
        self.bg         = []
        self.flow_type  = flow_type
        self.dists      = deque(maxlen=qlen)
    
    def get_bg(self,gray):
        self.bg           = self.bg if len(self.bg) else self.prev_gray
        diff              = cv2.absdiff(self.bg, gray)
        ret, motion_mask  = cv2.threshold(diff, self.thresh, self.assign_value, cv2.THRESH_BINARY)
        bg = self.alpha * gray + (1 - self.alpha) * self.bg
        self.bg = np.uint8(bg)  
        return motion_mask
    
    def get_pts(self,mask):
        cnts,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        coords = []
        mts    = []
        res    = mask * 0.0
        for c in cnts:
            M = cv2.moments(c)
            if M["m00"]>25:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                mts.append(-1*M["m00"])
                coords.append([[cX,cY]])
                #cnv1=cv2.drawContours(cnv1, [c], -1, (0, 255, 0), 10)
                res=cv2.circle(res, (cX, cY), 5, 255, -1)
                #cnv3=cv2.putText(cnv3, "center", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        coords = np.array(coords,dtype=np.float32)[np.argsort(mts)]
        return coords,res

    def get_flow(self,frame):
        #motion_mask1 = self.fgbg.apply(frame, -1)
        #background1  = self.fgbg.getBackgroundImage()
        flow_img = cnv  = frame
        if len(self.prev_frame) == 0:
            self.prev_gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.p0          = cv2.goodFeaturesToTrack(self.prev_gray, mask = None, **self.feature_params)
            self.prev_frame  = frame.copy()
            return frame,frame,self.dists
        
        #if len(self.p0)<20:
        gray        = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mt_mask     = self.get_bg(gray)
        pts,pt_mask = self.get_pts(mt_mask)
        if len(pts):
            self.p0 = np.concatenate([self.p0,pts],dtype=np.float32)[:25]
        cnv         = frame.copy()
        cnv[:,:,1]  = 0.8*cnv[:,:,1] + 0.2*mt_mask
        cnv[:,:,0]  = 0.8*cnv[:,:,0] + 0.2*pt_mask
        
        if self.flow_type=='dense':
            hsv        = np.zeros_like(frame)
            hsv[...,1] = 255
            flow       = cv2.calcOpticalFlowFarneback(self.prev_gray,gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang   = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            flow_img   = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        
        x,y,w,h  = int(pts[0][0]) , int(pts[0][1]), 25, 25
        roi      = frame[y:y+h, x:x+w]
        hsv_roi  = cv2.cvtColor(roi, cv.COLOR_BGR2HSV)
        mask     = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        cv2.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)
        
        hsv = cv2.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        ret, track_window = cv2.camShift(mt_mask, pts[0], term_crit)
        res               = cv2.circle(res, (cX, cY), 5, 255, -1)
        
        if self.flow_type=='sparse':
            #import pdb;pdb.set_trace()
            mask        = np.zeros_like(frame)
            if len(self.p0):
                p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.p0, None, **self.lk_params)
            
                if type(p1) != type(None):
                    # Select good points
                    good_new = p1[st==1]
                    good_old = self.p0[st==1]

                    # draw the tracks
                    dists=[]
                    for i,(new,old) in enumerate(zip(good_new,good_old)):
                        a,b   = new.ravel().astype('int32')
                        c,d   = old.ravel().astype('int32')
                        dists+= [np.sqrt((a-c)**2+(b-d)**2)]
                        mask  = cv2.line(mask, (a,b),(c,d), self.color[i].tolist(), 2)
                        frame = cv2.circle(frame,(a,b),5, self.color[i].tolist(),-1)
                    flow_img  = cv2.add(frame,mask)        
                    self.p0         = good_new.reshape(-1,1,2)
                    if not len(dists): dists=[0.0]
                    self.dists.append([np.max(dists),np.mean(dists)])
            #else:
            #    flow_img = frame
            #    self.dists.append([0.0,0.0])
            # Now update the previous frame and previous points
        self.prev_gray  = gray.copy()
        self.prev_frame = frame
        return flow_img,cnv,np.array(self.dists)