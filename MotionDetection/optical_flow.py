import cv2
import numpy as np
from copy import deepcopy
from collections import deque
from matplotlib import colors
import torch
from torchvision.ops import box_iou, nms

import sys
sys.path.append('../pytorch-openpose/')
from posedet_mnet.detector import PoseDet
from posedet_alpha.detector_alpha import PoseDet

posedet = PoseDet(size=480,cpu=False)

def hex2rgb(_hex):
    _hex = _hex.lstrip('#')
    return [int(_hex[i:i+2],16) for i in (0,2,4)]
colors = [hex2rgb(val) for val in colors.CSS4_COLORS.values()]


class KptTracker():
    def __init__(self,kpts_per_obj=18,max_no_objs=25,channels=2):
        self.max_no_objs = max_no_objs
        self.kpts_per_obj= kpts_per_obj
        self.channels    = channels
        self.queue       = deque(maxlen=500)
        
    def add(self,pts,nth):
        if nth==0:
            self.curr_frame      = np.zeros((self.kpts_per_obj,self.max_no_objs,self.channels),dtype=np.int32)
        self.curr_frame[:,nth,:] = pts
        self.queue.append(self.curr_frame)
    def get(self):
        return np.array(self.queue)
    
class OptFlow():
    def __init__(self,flow_type='sparse',qlen=125):
        
        self.feature_params = dict( maxCorners = 100,
                               qualityLevel = 0.3,
                               minDistance = 7,
                               blockSize = 7 )

        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (15,15),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.fgbg      = cv2.createBackgroundSubtractorMOG2(history=50, #200
                                                           varThreshold=32) #16
        
        self.background = None
        self.max_frames = 1000
        self.diff_thresh= 60
        self.alpha      = 0.1
        
        # Create some random colors
        self.colors     = colors#np.random.randint(0,255,(100,3))
        self.prev_frame = []
        self.bg         = []
        self.flow_type  = flow_type
        self.dists      = deque(maxlen=qlen)
        self.kpts       = KptTracker()
        self.methods    = methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
        
        self.trackers   = ['MIL','KCF','GOTURN','MOSSE']
        #self.tracker    = eval(f"cv2.Tracker{self.trackers[0]}_create")(self.trackers[0])
        
        self.font       = cv2.FONT_HERSHEY_SIMPLEX
        
    def init(self):
        self.prev_gray   = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.p0          = cv2.goodFeaturesToTrack(self.prev_gray, mask = None, **self.feature_params) #https://docs.opencv.org/3.4/d4/d8c/tutorial_py_shi_tomasi.html
        _,kps_mask,_,kps,poses = posedet.detect(self.frame)
        self.prev_frame  = self.frame.copy()
        self.flow_mask   = np.zeros_like(self.frame)
        self.prev_poses  = poses
        return self.frame,self.frame,self.dists
    
    def hog(img):
        gx = cv.Sobel(img, cv.CV_32F, 1, 0)
        gy = cv.Sobel(img, cv.CV_32F, 0, 1)
        mag, ang = cv.cartToPolar(gx, gy)
        bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
        bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)     # hist is a 64 bit vector
        return hist
    
    def get_bg(self,gray):
        self.bg           = self.bg if len(self.bg) else self.prev_gray
        diff              = cv2.absdiff(self.bg, gray)
        ret, motion_mask  = cv2.threshold(diff, self.diff_thresh, 255, cv2.THRESH_BINARY)
        bg = self.alpha * gray + (1 - self.alpha) * self.bg
        self.bg = np.uint8(bg)  
        return motion_mask
    
    def mask_to_pts(self,mask):
        kernel = np.ones((10,10),np.float32)
        mask   = cv2.erode(mask,np.ones((5,5),np.float32),0)
        mask   = cv2.dilate(mask,kernel,0)
        mask   = cv2.dilate(mask,kernel,0)
        cnts,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        pts    = []
        for c in cnts:
            M = cv2.moments(c)
            if M["m00"]>25:
                cX  = int(M["m10"] / M["m00"])
                cY  = int(M["m01"] / M["m00"])
                ofs = int(np.sqrt(M["m00"]))
                pts.append([cX,cY,ofs])
        return pts
    
    def refine_pts(self,pts):
        coords = []
        mts    = []
        for cX,cY,ofs in pts:
            roi = self.prev_gray[cX-ofs:cX+ofs,cY-ofs:cY+ofs]
            pts = cv2.goodFeaturesToTrack(roi, mask = None, **self.feature_params) 
            #fast = cv2.FastFeatureDetector_create() # https://docs.opencv.org/4.x/df/d0c/tutorial_py_fast.html
            pts = None
            if type(pts) != type(None):
                for pt in pts:
                    _ofs = max(ofs,5)
                    _cX,_cY = int(pt[0][0]+(cX-_ofs)),int(pt[0][1]+(cY-_ofs))
                    coords.append([[_cX,_cY]])
                    mts.append(-1*ofs)
                    self.cnv=cv2.circle(self.cnv, (_cX, _cY), 5, 255, -1)
            else:
                coords.append([[cX,cY]])
                mts.append(-1*ofs)
                self.cnv=cv2.circle(self.cnv, (cX, cY), 3, 255, -1)

            #cnv1=cv2.drawContours(cnv1, [c], -1, (0, 255, 0), 10)
            #cnv3=cv2.putText(cnv3, "center", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if len(mts):
            args   = np.argsort(mts)
            #for i,arg in enumerate(args[:len(colors)]):
            #    self.cnv = cv2.drawContours(self.cnv, [cts[arg]], 0, colors[i], 3)
            coords = np.array(coords,dtype=np.float32)[args]
        return coords
    
    def get_cam_shift(self):
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
        
    def get_dense_flow(self):
        if self.flow_type=='dense':
            hsv        = np.zeros_like(self.frame)
            hsv[...,1] = 255
            flow       = cv2.calcOpticalFlowFarneback(self.prev_gray,self.gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang   = cv2.cartToPolar(flow[...,0], flow[...,1])
            self.cnv   = cv2.putText(self.cnv, f"{mag.max()}", (20, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            #mag        = 2.5* np.clip(mag,1,100)#
            mag        = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = mag
            
            flow_img   = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            #self.cnv[:,:,1] = 0.2*self.cnv[:,:,1] + 0.8*hsv[...,2]
    
    def get_sparse_flow(self,p0):     
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, self.gray, p0, None, **self.lk_params)
        if type(p1) != type(None):
            dists = np.zeros_like(p0)
            p2    = -1*np.ones_like(p0)
            for i,(_p0,_p1,_st) in enumerate(zip(p0,p1,st)):
                if _st!=1: continue
                p2[i] = p1[i]
            if not len(dists): dists=[0.0]
            self.dists.append([np.max(dists),np.mean(dists)])
        
        acc = sum([p[0][0]>0 for p in p1])/sum([p[0][0]>0 for p in p0])  
        return p2,round(acc,2)
        
    def run(self,frame):
        
        flow_img   = frame
        self.frame = cv2.GaussianBlur(frame,(5,5),0)
        self.cnv   = frame.copy()
        self.gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if len(self.prev_frame) == 0: return self.init()
            
        _,kps_mask,_,kps,poses = posedet.detect(frame)
        curr_boxes = torch.tensor([pose['bbox'] for pose in poses])
        #pts = self.refine_pts(pts)
        
        saved_prev_poses = deepcopy(self.prev_poses)
        for i in range(len(self.prev_poses)):
            prev_pose    = deepcopy(self.prev_poses[i])
            self.kpts.add(prev_pose['pts'].reshape(-1,2),i)
            prev_pose['pts'],acc = self.get_sparse_flow(prev_pose['pts'])
            '''
            if len(curr_boxes):
                valid_pts    = np.array([elem for elem in prev_pose[0] if elem[0][0]!=-1 and elem[0][1]!=-1])
                x,y,w,h      = cv2.boundingRect(valid_pts)
                prev_pose[1] = [int(x),int(y),int(x+w),int(y+h)]
                prev_box     = torch.tensor(prev_pose[1])[None]                        
                ious = box_iou(prev_box,curr_boxes)
                if ious.max()<0.3: continue
                curr_pose = poses[ious.argmax()]
                for j,[pt1,pt2] in enumerate(zip(curr_pose[0],prev_pose[0])):
                    if pt1[0][0] == -1: continue
                    if pt2[0][0] == -1: prev_pose[0][j][0]=curr_pose[0][j][0]
                    else: 
                        c1 = curr_pose[2]
                        c2 = prev_pose[2]
                        prev_pose[0][j] = (1/(c1+c2))*(c1*prev_pose[0][j] + c2*curr_pose[0][j])
                    
            '''
            dists=[]
            for j in range(len(prev_pose['pts'])):
                a,b = self.prev_poses[i]['pts'][j].reshape(2).astype('int32')
                c,d = prev_pose['pts'][j].reshape(2).astype('int32')
                if a>0 and c>0 : 
                    dists.append(np.sqrt((a-c)**2+(b-d)**2))
                    self.flow_mask  = cv2.line(self.flow_mask, (a,b),(c,d), self.colors[i], 1)
            text = f"Acc : {acc} Vel : {round(np.mean(dists),2)}"
            (tw,th) = cv2.getTextSize(text,self.font,0.4,1)[0]
            self.cnv  = cv2.putText(self.cnv, text, (20, 20+(th+3)*(i+1)),self.font, 0.4, self.colors[i], 1)
            #    #self.flow_mask  = cv2.circle(self.flow_mask,(a,b),5, self.color[i].tolist(),-1)
            self.prev_poses[i] = prev_pose
        '''
        for j,[pose1,pose2] in enumerate(zip(self.prev_poses,saved_prev_poses)):
            dists = [0.0]*len(pose1[0])
            for pt1,pt2 in zip(pose1[0],pose2[0]):
                a,b      = pt1[0]
                c,d      = pt2[0]
                
                #dists[i] = [np.sqrt((a-c)**2+(b-d)**2)]
                self.flow_mask  = cv2.line(self.flow_mask, (a,b),(c,d), self.color[i].tolist(), 2)
                #self.flow_mask  = cv2.circle(self.flow_mask,(a,b),5, self.color[i].tolist(),-1)   
        '''
        self.prev_gray  = self.gray
        self.prev_frame = self.frame
        self.prev_poses = poses if len(poses)>len(self.prev_poses) else self.prev_poses
        
        flow_img  = cv2.add(self.frame,self.flow_mask)    
        return flow_img,self.cnv,np.array(self.dists)
    '''
    def run_v1(self,frame):
        
        flow_img = frame
        self.frame = cv2.GaussianBlur(frame,(5,5),0)
        self.cnv   = frame.copy()
        self.gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if len(self.prev_frame) == 0: return self.init()
            
        #mt_mask      = self.fgbg.apply(frame, -1)
        #mt_mask     = self.get_bg(gray)
        #res = cv.matchTemplate(img,template,method)
        
        _,kps_mask,_,kps,poses = posedet.detect(frame)
        #pts = np.array([[int(kp[0]),int(kp[1]),int(10*kp[2])] for kp in kps if kp[2]>0.2])
        curr_poses = [pose[0] for pose in poses] 
        prev_boxes = torch.tensor([pose[1] for pose in self.prev_poses])
        curr_boxes = torch.tensor([pose[1] for pose in poses])
        
        #pts = self.mask_to_pts(mt_mask)
        #pts = self.refine_pts(pts)
        for i in range(len(self.prev_poses)):
            pts       = self.get_sparse_flow(self.prev_poses[i][0])
            prev_pose = self.prev_poses[i]
            prev_box  = prev_pose[1][None]
            ious      = box_iou(prev_box,curr_boxes)
            if ious.max()>0.5:
                curr_pose = poses[ious.argmax()]
            for j,[pt1,pt2] in enumerate(zip(curr_pose,prev_pose)):
                if pt1[0] == -1: continue
                if pt2[0] == -1: prev_pose[j]=curr_pose[j]
                else: prev_pose[j] = 0.5*(prev_pose[j] + curr_pose[j])
            
            self.prev_poses[i][0] = pts
        #if len(pts)> len(self.p0):
        #    self.p0 = pts #pts#np.concatenate([self.p0,pts],dtype=np.float32)[:25]
        
        #cnv[:,:,1]  = 0.8*cnv[:,:,1] + 0.2*mt_mask
        #self.cnv[:,:,0]  = 0.2*self.cnv[:,:,0] + 0.8*pt_mask
        
        self.prev_gray  = self.gray
        self.prev_frame = self.frame
        self.prev_poses = poses if len(poses)>len(self.prev_poses) else self.prev_poses
        
        flow_img  = cv2.add(self.frame,self.flow_mask)    
        return flow_img,self.cnv,np.array(self.dists)
        '''