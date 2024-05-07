import cv2
import numpy as np
class GUI():
    def __init__(self,H=800,W=1200,n_view=4,color=(0,0,0)):
        self.ofs         = 20
        self.win         = np.ones((H,W,3),dtype=np.uint8)
        self.win         *=np.uint8(color)
        self.view_coords = list([[j,i] for i in range(0,H,H//n_view) for j in range(0,W,W//n_view)])
        self.n_view      = n_view
        self.view_hw     = H//n_view, W//n_view
        for x in range(0,H,H//n_view):
            cv2.line(self.win,(0,x),(W,x),(100,100,100),5)
        for y in range(0,W,W//n_view):
            cv2.line(self.win,(y,0),(y,H),(100,100,100),5)

    def add_view(self,view,win_id,ofs=10):
        x,y = self.view_coords[win_id]
        x   += ofs
        y   += ofs
        h,w = self.view_hw
        h   -= 2*ofs
        w   -= 2*ofs
        self.win[y:y+h,x:x+w] = cv2.resize(view,(w,h))
        return self.win

