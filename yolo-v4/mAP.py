#30/03/2022  
#================================================================
# Yolo V-3 by irfanhasib.me@gmail.com
# Inspired by -
# GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#================================================================

import os
import sys 
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED']=str(0)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import glob
import time
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm,trange

path = './cocoapi/PythonAPI/build/lib.linux-x86_64-3.8/'
sys.path.insert(0,path)
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from yolo.model import YoloModel, calc_yolo_loss, calc_seg_loss
from yolo.decoder import YoloDecodeNetout
from yolo.dataset import Dataset
from yolo.eval import get_mAP
from yolo.utils import Utils
from yolo.seg_loader import Seg_Utils
from yolo.config import *
from yolo.tf import *

epoch        = 15
input_size   = 320
exp_log_path = ''#'exp-MNET_V2_320_IND_539' #[15 , 19.1] #[
#exp_log_path = 'logs/_exp-MNET_V2_224_V32_MUL_0001_EXP_179/' # 10
task = 'pred'

if __name__ == '__main__' :
    epoch        = sys.argv[1]
    input_size   = int(sys.argv[2])
    exp_log_path = sys.argv[3]
    task = sys.argv[4]
    os.system(f'cp {DATA_DIR}/COCO/annotations_trainval2017/annotations/instances_val2017.json ./logs/{exp_log_path}/gt.json')
    print(f"Running {task} for epoch : {epoch} input_size : {input_size} ; {exp_log_path}")
    #sys.exit(0)
    if task == 'pred':
        yolo = YoloModel()
        yolo_model = yolo.get_model()
        yolo_model.load_weights(glob.glob(f'./logs/{exp_log_path}/model/epoch_{epoch}*')[0]+'/weights')
        decoder = YoloDecodeNetout()

        for i,layer in enumerate(yolo_model.layers):
            yolo_model.get_layer(layer.name).trainable=False

        with open(f'./logs/{exp_log_path}/gt.json','r') as file:
             val_data = json.load(file)
        img_root = 'COCO/val2017/'

        gen_preds= True
        rets = {}
        for score_threshold,iou_threshold in zip([0.01],[0.5]):
            print(f'Calculation mAP for score_threshold : {score_threshold} ,iou_threshold : {iou_threshold}')

            if gen_preds :
                with open(f'logs/{exp_log_path}/gt.json','r') as file:
                     val_data = json.load(file)


                results =[]
                for img in tqdm(val_data['images']):
                    results+= decoder.detect_image(yolo_model,yolo.decode_output, img, root=RAW_DATA_DIR+img_root, output_path=TRAIN_CHECKPOINTS_FOLDER+'/pred_imgs/',input_size=input_size, show=True, score_threshold=score_threshold, iou_threshold=iou_threshold, rectangle_colors='',draw=False)

                with open(f'logs/{exp_log_path}/pred_{score_threshold}_{iou_threshold}_{epoch}.json','w') as file:
                    json.dump(results,file)
        del yolo_model, decoder, results
    if task == 'calc':
        
        score_threshold,iou_threshold =0.01,0.5
        gt_file = f'logs/{exp_log_path}/gt.json'
        gt = COCO(gt_file)
        pred = gt.loadRes(f'logs/{exp_log_path}/pred_{score_threshold}_{iou_threshold}_{epoch}.json')

        cocoEval = COCOeval(gt,pred,iouType='bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        save_path = f'results/{exp_log_path}_pred_{score_threshold}_{iou_threshold}.pkl'
        if os.path.exists(save_path):
            with open(save_path,'rb') as file:
                rets = pickle.load(file)
                rets[epoch] = cocoEval.stats
        else:
            rets={epoch:cocoEval.stats}
        with open(save_path,'wb') as file:
            pickle.dump(rets,file)

        del gt,pred, cocoEval