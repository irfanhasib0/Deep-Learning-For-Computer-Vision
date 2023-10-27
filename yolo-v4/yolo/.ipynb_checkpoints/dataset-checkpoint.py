from .config import *
from .utils import Utils
from .seg_loader import Seg_Loader
#from .tf import *
import numpy as np
np.random.seed(SEED)
import cv2
import os
import random
from collections import defaultdict
import json

random.seed(SEED)
ANCHORS = np.array(ANCHORS, np.float32)/416.0
    
class Dataset(object):
    # Dataset preprocess implementation
    def __init__(self, dataset_type, input_size = TRAIN_INPUT_SIZE):
        self.annot_path  = TRAIN_ANNOT_PATH if dataset_type == 'train' else TEST_ANNOT_PATH
        self.input_sizes = input_size if dataset_type == 'train' else input_size
        self.batch_size  = TRAIN_BATCH_SIZE if dataset_type == 'train' else TEST_BATCH_SIZE
        self.data_aug    = TRAIN_DATA_AUG   if dataset_type == 'train' else TEST_DATA_AUG
        self.root        = TRAIN_IMG_PATH   if dataset_type == 'train' else TEST_IMG_PATH

        self.train_input_size   = input_size
        self.strides            = np.array(YOLO_STRIDES)
        self.classes            = CLASS_NAMES
        self.num_classes        = len(self.classes)
        self.anchors            = ANCHORS
        self.anchor_per_scale   = YOLO_ANCHOR_PER_SCALE
        self.max_bbox_per_scale = YOLO_MAX_BBOX_PER_SCALE

        self.annotations        = self.load_annotations_coco(self.annot_path)
        self.num_samples        = len(self.annotations)
        self.num_batchs         = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count        = 0
        if TRAIN_USE_SEG  == True or TRAIN_USE_DST  == True:
            self.seg_loader = Seg_Loader(annot_path = self.annot_path.replace('txt','json'))
        
    def load_annotations(self, dataset_type):
        final_annotations = []
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        #np.random.shuffle(annotations)
        
        for annotation in annotations:
            # fully parse annotations
            line = annotation.split()
            image_path, H,W = line[0].split(',')
            
            if not os.path.exists(self.root+image_path):
                raise KeyError(f"Img does not exist ... {self.root}{image_path}")
            if TRAIN_LOAD_IMAGES_TO_RAM:
                image = cv2.imread(self.root+image_path)
            else:
                image = ''
            final_annotations.append([str(image_path), line[1:], image])
        return final_annotations
    
    def parse_annotation(self, annotation, mAP = False):
        if TRAIN_LOAD_IMAGES_TO_RAM:
            image_path = annotation[0]
            image = annotation[2]
        else:
            image_path = annotation[0]
            image = cv2.imread(self.root+image_path)
        image = np.array(image,dtype=np.float32)   
        bboxes = np.array([list(map(float, box.split(','))) for box in annotation[1]])

        if self.data_aug:
            image, bboxes = Utils.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = Utils.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = Utils.random_translate(np.copy(image), np.copy(bboxes))

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if mAP == True: 
            #image = cv2.imread(self.root+image_path)
            return self.root+image_path, bboxes
        
        image, bboxes = Utils.image_preprocess(image, [self.input_sizes, self.input_sizes], np.copy(bboxes))
        return image, bboxes
    
    def get_cat_map(self,categories):
        class_dict = {val:key for key,val in CLASS_NAMES.items()}
        id_dict = {}
        for cat in categories:
            name = cat['name']
            if name == 'motorcycle' : name = 'motorbike'
            if name == 'airplane' : name = 'aeroplane'
            if name == 'couch' : name = 'sofa'
            if name == 'tv' : name = 'tvmonitor'
            try : id_dict [cat['id']] = class_dict[name]
            except : 
                try : id_dict [cat['id']] = class_dict[name.replace(' ','-')]
                except : id_dict [cat['id']] = class_dict[name.replace(' ','')]
        #rev_id_dict = {val:key for key,val in id_dict.items()}
        return id_dict
    
    def load_annotations_coco(self,annot_path):
        with open(annot_path.replace('.txt','.json'),'r') as file:
             data = json.load(file)
        images      = data['images']
        annotations = data['annotations']
        categories  = data['categories']

        id_dict = self.get_cat_map(categories)
        annot_dict = defaultdict(list)
        for annot in annotations:
            annot_dict[annot['image_id']].append({'bbox':annot['bbox'], 'category_id' : id_dict[annot['category_id']]})

        label_dict = {}
        for img in images:
            _id       = img['id']
            file_name = img['file_name']
            height    = img['height']
            width     = img['width']
            annots    = annot_dict[_id]
            _annots = []
            for ann in annots:
                ann['bbox'] = [ann['bbox'][0] + (0.5 * ann['bbox'][2]), ann['bbox'][1] + (0.5 * ann['bbox'][3]), ann['bbox'][2], ann['bbox'][3]]
                ann['bbox'] = [ann['bbox'][0]/width, ann['bbox'][1]/height, ann['bbox'][2]/width, ann['bbox'][3]/height]
                _annots+=[ann]
            label_dict[_id] = {'file_name':file_name,'height':height,'width':width,'annotations':_annots}
        return list(label_dict.values())
    
    def parse_annotations_coco(self,annots):
        bboxes = np.array([annot['bbox']+[annot['category_id']] for annot in annots['annotations']],np.float32)
        image = cv2.imread(self.root+annots['file_name'])
        image, bboxes = Utils.image_preprocess(image, [self.input_sizes, self.input_sizes], np.copy(bboxes))
        return image,bboxes
    
    def __next__(self):
        #with tf.device('/cpu:0'):
            self.train_output_sizes = self.train_input_size // self.strides

            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3), dtype=np.float32)

            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)

            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_seg_lbls= np.zeros((self.batch_size, TRAIN_INPUT_SIZE, TRAIN_INPUT_SIZE, len(CLASS_NAMES)),dtype = np.uint8)
            num = 0
            
            while num < self.batch_size:
                index = self.batch_count * self.batch_size + num
                if index >= self.num_samples: 
                    index -= self.num_samples

                annotation = self.annotations[index]
                if TRAIN_USE_SEG  == True or TRAIN_USE_DST  == True:
                    batch_seg_lbls[num]  = self.seg_loader.get_seg_masks(annotation[0])

                image, bboxes = self.parse_annotations_coco(annotation)
                label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)

                batch_image[num, :, :, :] = image
                batch_label_sbbox[num, :, :, :, :] = label_sbbox
                batch_label_mbbox[num, :, :, :, :] = label_mbbox
                batch_label_lbbox[num, :, :, :, :] = label_lbbox
                batch_sbboxes[num, :, :] = sbboxes
                batch_mbboxes[num, :, :] = mbboxes
                batch_lbboxes[num, :, :] = lbboxes
                num += 1

            batch_smaller_target = [batch_label_sbbox, batch_sbboxes]
            batch_medium_target  = [batch_label_mbbox, batch_mbboxes]
            batch_larger_target  = [batch_label_lbbox, batch_lbboxes]
            
            self.batch_count += 1
            if self.batch_count >= self.num_batchs:
                self.batch_count = 0
                
            if TRAIN_USE_SEG  == True or TRAIN_USE_DST  == True:
                return batch_image, [batch_smaller_target, batch_medium_target, batch_larger_target, batch_seg_lbls]
            else:
                return batch_image, [batch_smaller_target, batch_medium_target, batch_larger_target][:NO_OF_GRID]
            
        
    def preprocess_true_boxes(self, bboxes):
        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale, 5 + self.num_classes)) for i in range(3)]
        
        bboxes_xywh = np.zeros((3,self.max_bbox_per_scale, 4))
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_xywh = bbox[:4]
            bbox_class_ind = int(bbox[4])
            
            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            #bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            #print(bbox_xywh)
            iou = []
            exist_positive = False
            bboxes_xywh_scl = []
            
            for i in range(NO_OF_GRID):
                grid_w = grid_h      = self.train_input_size/self.strides[i]
                bbox_xywh_scl        = bbox_xywh * grid_w # i= 0 > 416/16 = 14 ; i= 1 > 416/32 = 7 
                anchors_xywh         = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scl[0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i] * grid_w
                
                b1 = Utils.c_xywh2xyxy(bbox_xywh_scl.reshape(1,-1).copy())
                b2 = Utils.c_xywh2xyxy(anchors_xywh.copy())
                
                #print('b1\n',bbox_xywh_scl,'\nb2\n',anchors_xywh)
                iou_scale = Utils.bboxes_iou(b1,b2 )
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3
                
                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scl[0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask,  : ]  = 0
                    label[i][yind, xind, iou_mask, 0:4]  = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5]  = 1.0
                    label[i][yind, xind, iou_mask, 5: ]  = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i,bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True
                bboxes_xywh_scl +=[bbox_xywh_scl]
            #print(self.train_input_size,self.strides[i],grid_w,bboxes_xywh_scl)
            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect     = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor     = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind      = np.floor(bboxes_xywh_scl[best_detect][0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor,  : ] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5: ] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect,bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
            #print(iou)
        label_sbbox, label_mbbox, label_lbbox = label[0],label[1],label[2]
        sbboxes, mbboxes, lbboxes = bboxes_xywh[0],bboxes_xywh[1],bboxes_xywh[2]
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
    
    
    def __iter__(self):
        return self
  
    def __len__(self):
        return self.num_batchs
