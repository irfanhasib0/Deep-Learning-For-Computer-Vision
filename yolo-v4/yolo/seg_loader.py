import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
from collections import defaultdict
import json
import zlib
from .tf import *
from yolo.config import *

class Seg_Loader():
    def __init__(self,annot_path ='D:\\COCO\\annotations_trainval2017\\annotations\\instances_train2017.json'):
        
        with open(annot_path,'r') as file:
            self._dict=json.load(file)
        
        self.img_dict={}
        self.path_to_id ={}
        for img in self._dict['images']:
            self.img_dict[img['id']] = img
            self.path_to_id[img['file_name']] = img['id']
                            
        self.annot_dict=defaultdict(list)
        for annot in self._dict['annotations']:
            _id = annot['image_id']
            self.annot_dict[_id].append(annot)

        self.cat_dict={}
        for ind,cat in enumerate(self._dict['categories']):
            self.cat_dict[cat['id']]=[cat['name'],cat['supercategory'],ind]
        
    def rle_decode(self,values,size):
        data=[]
        v = 0
        for val in values:
            data+=[v]*val
            v = abs(v-255)
        return np.array(data,dtype=np.uint8).reshape((size[1],size[0])).T

    def decode_mask(self,segs,H,W):

        if type(segs) == list:
            mask = np.zeros((H,W),dtype=np.uint8)
            for seg in segs:
                seg = np.array(seg)
                pts=[]
                pt=[]
                for i,point in enumerate(seg):
                    pt.append(point)
                    if (i+1)%2==0: 
                        pts.append(pt)
                        pt=[]
                _pts = np.array(pts,np.int32).reshape(-1,2)
                mask = cv2.fillPoly(mask,[_pts],(255))

        elif type(segs) == dict:
            values=segs['counts']
            size=segs['size']
            mask = self.rle_decode(values,size)
        return mask

    def get_seg_masks(self,img_path,vis=False):
        img_id = self.path_to_id[img_path]
        annots = self.annot_dict[img_id]
        H     = self.img_dict[img_id]['height']
        W     = self.img_dict[img_id]['width']
        fname = self.img_dict[img_id]['file_name']
            
        seg_label = np.zeros((TRAIN_INPUT_SIZE,TRAIN_INPUT_SIZE,len(self.cat_dict)),dtype=np.uint8)
        
        for annot in annots:
            seg = annot['segmentation']
            cat_name = self.cat_dict[annot['category_id']][0]
            cat_ind = self.cat_dict[annot['category_id']][2]
        
            
            mask=self.decode_mask(seg,H,W)
            mask = cv2.resize(mask,(TRAIN_INPUT_SIZE,TRAIN_INPUT_SIZE))
            seg_label[:,:,cat_ind]+=np.array(mask/255,dtype=np.uint8)
            
            if vis:
                img_root='D:\\COCO\\train2017\\'
                img_path = img_root + fname
                img = cv2.imread(img_path)
                img = cv2.resize(img,(TRAIN_INPUT_SIZE,TRAIN_INPUT_SIZE))
                fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(10,4))
                axes[0].imshow(0.9*img[:,:,0]+mask,cmap='gray') 
                axes[1].imshow(img)
                plt.show()

        return seg_label



class Seg_Utils():
    def __init__(self):
        json_path="D:/COCO/annotations_trainval2017/annotations/instances_train2017.json"
        with open(json_path,'r') as file:
            val_gt=json.load(file)
        self.categories = val_gt['categories']
        del val_gt
        
        self.super_cats = defaultdict(list)
        for ind,cat in enumerate(self.categories):
            self.super_cats[cat['supercategory']].append(ind)
    
    def load_seg(self,batch_seg,bsize=TEST_BATCH_SIZE):
        batch_seg = np.frombuffer(zlib.decompress(batch_seg),dtype=np.uint8).reshape(bsize,TRAIN_INPUT_SIZE//TRAIN_SEG_SCALE,TRAIN_INPUT_SIZE//TRAIN_SEG_SCALE)
        
        #if TRAIN_SEG_SUP_CAT:
        #    batch_seg = self.conv_to_sup_cats(batch_seg)
        
        
        return [batch_seg]
    
    def conv_to_sup_cats(self,label):
        nshape = label.shape[:-1]+(len(self.super_cats),)
        nlabel = np.zeros(nshape,dtype=np.uint8)
        for ind,cat in enumerate(self.super_cats.keys()):
            nlabel[:,:,:,ind] = label[:,:,:,self.super_cats[cat]].max(axis=-1)
        return nlabel
    
    