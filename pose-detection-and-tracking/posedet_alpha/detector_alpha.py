import os
import sys
import torch
import platform
import math
import time

import cv2
import numpy as np

from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.utils.pPose_nms import pose_nms
from alphapose.utils.presets import SimpleTransform, SimpleTransform3DSMPL
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.models import builder
from alphapose.utils.config import update_config
from detector.apis import get_detector
from alphapose.utils.vis import getTime

# https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth
#https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
#https://github.com/Megvii-BaseDetection/YOLOX?tab=readme-ov-file

# ResNet50 MAP72 https://drive.google.com/open?id=1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn

#https://drive.usercontent.google.com/u/0/uc?id=1myNKfr2cXqiHZVXaaG8ZAq_U2UpeOLfG&export=download

#! python3 scripts/demo_inference.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir examples/demo/ --save_img
#https://drive.usercontent.google.com/download?id=1Bb3kPoFFt-M0Y3ceqNO8DTXi1iNDd4gI&export=download&authuser=0
class Params():
    def __init__(self):                                        
        #self.cfg           = '../models/pose/alpha_pose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml'
        self.cfg           = '../models/pose/alpha_pose/configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml'
        #self.cfg           = '../models/pose/cfg/256x192_res50_lr1e-3_2x-regression.yaml'
        #self.checkpoint    = '../models/pose/multi_domain_fast50_regression_256x192_2.pth'
        self.checkpoint    = '../models/pose/alpha_pose/fast_421_res152_256x192.pth'
        self.detector      = 'yolox_x'
        self.image         = ''
        self.save_img      = False
        self.vis           = True
        self.showbox       = True
        self.profile       = False
        self.format        = 'coco' #'cmu' / 'open'
        self.min_box_area = 0
        self.eval         = False
        self.gpus         = '0'
        self.flip         = False
        self.debug        = False
        self.vis_fast     = False
        self.pose_flow    = False
        self.pose_track   = False
        self.device       = 'cuda'
        
        self.gpus = [int(self.gpus[0])] if torch.cuda.device_count() >= 1 else [-1]
        self.device = torch.device("cuda:" + str(self.gpus[0]) if self.gpus[0] >= 0 else "cpu")
        self.tracking = self.pose_track or self.pose_flow or self.detector=='tracker'

class DetectionLoader():
    def __init__(self, detector, cfg, opt):
        self.cfg = cfg
        self.opt = opt
        self.device = opt.device
        self.detector = detector

        self._input_size = cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = cfg.DATA_PRESET.HEATMAP_SIZE

        self._sigma = cfg.DATA_PRESET.SIGMA

        if cfg.DATA_PRESET.TYPE == 'simple':
            pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
            self.transformation = SimpleTransform(
                pose_dataset, scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0, sigma=self._sigma,
                train=False, add_dpg=False, gpu_device=self.device)
        elif cfg.DATA_PRESET.TYPE == 'simple_smpl':
            # TODO: new features
            from easydict import EasyDict as edict
            dummpy_set = edict({
                'joint_pairs_17': None,
                'joint_pairs_24': None,
                'joint_pairs_29': None,
                'bbox_3d_shape': (2.2, 2.2, 2.2)
            })
            self.transformation = SimpleTransform3DSMPL(
                dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
                color_factor=cfg.DATASET.COLOR_FACTOR,
                occlusion=cfg.DATASET.OCCLUSION,
                input_size=cfg.MODEL.IMAGE_SIZE,
                output_size=cfg.MODEL.HEATMAP_SIZE,
                depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
                bbox_3d_shape=(2.2, 2,2, 2.2),
                rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
                train=False, add_dpg=False, gpu_device=self.device,
                loss_type=cfg.LOSS['TYPE'])

        self.image = (None, None, None, None)
        self.det = (None, None, None, None, None, None, None)
        self.pose = (None, None, None, None, None, None, None)

    def process(self,image):
        # start to pre process images for object detection
        self.image_preprocess(image)
        # start to detect human in images
        self.image_detection()
        # start to post process cropped human image for pose estimation
        self.image_postprocess()
        return self

    def image_preprocess(self,image):
        # expected image shape like (1,3,h,w) or (3,h,w)
        img = self.detector.image_preprocess(image)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        # add one dimension at the front for batch if image shape (3,h,w)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        orig_img = image # scipy.misc.imread(im_name_k, mode='RGB') is depreciated
        im_dim = orig_img.shape[1], orig_img.shape[0]

        with torch.no_grad():
            im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

        self.image = (img, orig_img, im_dim)

    def image_detection(self):
        imgs, orig_imgs, im_dim_list = self.image
        if imgs is None:
            self.det = (None, None, None, None, None, None, None)
            return

        with torch.no_grad():
            dets = self.detector.images_detection(imgs, im_dim_list)
            if isinstance(dets, int) or dets.shape[0] == 0:
                self.det = (orig_imgs, None, None, None, None, None)
                return
            if isinstance(dets, np.ndarray):
                dets = torch.from_numpy(dets)
            dets = dets.cpu()
            boxes = dets[:, 1:5]
            scores = dets[:, 5:6]
            ids = torch.zeros(scores.shape)

        boxes = boxes[dets[:, 0] == 0]
        if isinstance(boxes, int) or boxes.shape[0] == 0:
            self.det = (orig_imgs, None, None, None, None, None)
            return
        inps = torch.zeros(boxes.size(0), 3, *self._input_size)
        cropped_boxes = torch.zeros(boxes.size(0), 4)

        self.det = (orig_imgs, boxes, scores[dets[:, 0] == 0], ids[dets[:, 0] == 0], inps, cropped_boxes)

    def image_postprocess(self):
        with torch.no_grad():
            (orig_img, boxes, scores, ids, inps, cropped_boxes) = self.det
            if orig_img is None:
                self.pose = (None, None, None, None, None, None)
                return
            if boxes is None or boxes.nelement() == 0:
                self.pose = (None, orig_img, boxes, scores, ids, None)
                return

            for i, box in enumerate(boxes):
                inps[i], cropped_box = self.transformation.test_transform(orig_img, box)
                cropped_boxes[i] = torch.FloatTensor(cropped_box)

            self.pose = (inps, orig_img, boxes, scores, ids, cropped_boxes)

    def read(self):
        return self.pose


class DataWriter():
    def __init__(self, cfg, opt):
        self.cfg = cfg
        self.opt = opt

        self.eval_joints = list(range(cfg.DATA_PRESET.NUM_JOINTS))
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        self.item = (None, None, None, None, None, None)
        
        loss_type = self.cfg.DATA_PRESET.get('LOSS_TYPE', 'MSELoss')
        num_joints = self.cfg.DATA_PRESET.NUM_JOINTS
        if loss_type == 'MSELoss':
            self.vis_thres = [0.4] * num_joints
        elif 'JointRegression' in loss_type:
            self.vis_thres = [0.05] * num_joints
        elif loss_type == 'Combined':
            if num_joints == 68:
                hand_face_num = 42
            else:
                hand_face_num = 110
            self.vis_thres = [0.4] * (num_joints - hand_face_num) + [0.05] * hand_face_num

        self.use_heatmap_loss = (self.cfg.DATA_PRESET.get('LOSS_TYPE', 'MSELoss') == 'MSELoss')

    def get_pose(self,boxes, scores, ids, hm_data, cropped_boxes, orig_img,expand_ratio=0.1):
        norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        hm_size   = self.cfg.DATA_PRESET.HEATMAP_SIZE
        if orig_img is None: return []
        else:  H,W,_ = orig_img.shape
        # image channel RGB->BGR
        orig_img = np.array(orig_img, dtype=np.uint8)[:, :, ::-1]
        self.orig_img = orig_img
        if boxes is None or len(boxes) == 0:
            return []
        else:
            # location prediction (n, kp, 2) | score prediction (n, kp, 1)
            assert hm_data.dim() == 4
            if hm_data.size()[1] == 136:
                self.eval_joints = [*range(0,136)]
            elif hm_data.size()[1] == 26:
                self.eval_joints = [*range(0,26)]
            elif hm_data.size()[1] == 133:
                self.eval_joints = [*range(0,133)]
            pose_coords = []
            pose_scores = []

            for i in range(hm_data.shape[0]):
                bbox = cropped_boxes[i].tolist()
                if isinstance(self.heatmap_to_coord, list):
                    pose_coords_body_foot, pose_scores_body_foot = self.heatmap_to_coord[0](
                        hm_data[i][self.eval_joints[:-110]], bbox, hm_shape=hm_size, norm_type=norm_type)
                    pose_coords_face_hand, pose_scores_face_hand = self.heatmap_to_coord[1](
                        hm_data[i][self.eval_joints[-110:]], bbox, hm_shape=hm_size, norm_type=norm_type)
                    pose_coord = np.concatenate((pose_coords_body_foot, pose_coords_face_hand), axis=0)
                    pose_score = np.concatenate((pose_scores_body_foot, pose_scores_face_hand), axis=0)
                else:
                    pose_coord, pose_score = self.heatmap_to_coord(hm_data[i][self.eval_joints], bbox, hm_shape=hm_size, norm_type=norm_type)
                pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
            preds_img = torch.cat(pose_coords)
            preds_scores = torch.cat(pose_scores)

            boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                pose_nms(boxes, scores, ids, preds_img, preds_scores, self.opt.min_box_area, use_heatmap_loss=self.use_heatmap_loss)

            result = []
            for k in range(len(scores)):
                w,h= (boxes[k][2] - boxes[k][0]), (boxes[k][3] - boxes[k][1])
                result.append(
                    {
                        'keypoints':preds_img[k].numpy().astype('int32'),
                        'kp_score':preds_scores[k].numpy(),
                        'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]).numpy(),
                        'idx':ids[k],
                        'bbox':np.array([np.clip(boxes[k][0] - expand_ratio*w,0,W), 
                                         np.clip(boxes[k][1] - expand_ratio*h,0,H), 
                                         np.clip(boxes[k][2] + expand_ratio*w,0,W), 
                                         np.clip(boxes[k][3] + expand_ratio*h,0,H)]) 
                    }
                )


            if hm_data.size()[1] == 49:
                from alphapose.utils.vis import vis_frame_dense as vis_frame
            elif self.opt.vis_fast:
                from alphapose.utils.vis import vis_frame_fast as vis_frame
            else:
                from alphapose.utils.vis import vis_frame
            self.vis_frame = vis_frame
        return result

class PoseDet():
    def __init__(self):
        self.args = Params()
        self.cfg  = update_config(self.args.cfg)
        # Load pose model
        self.pose_model = builder.build_sppe(self.cfg.MODEL, preset_cfg=self.cfg.DATA_PRESET)

        print(f'Loading pose model from {self.args.checkpoint}...')
        self.pose_model.load_state_dict(torch.load(self.args.checkpoint, map_location=self.args.device))
        self.pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)

        self.pose_model.to(self.args.device)
        self.pose_model.eval()
        
        self.det_loader = DetectionLoader(get_detector(self.args), self.cfg, self.args)

    def detect(self, img,show=False):
        img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Init data writer
        self.writer = DataWriter(self.cfg, self.args)

        runtime_profile = {
            'dt': [],
            'pt': [],
            'pn': []
        }
        pose = []
        
        start_time = getTime()
        with torch.no_grad():
            (inps, orig_img, boxes, scores, ids, cropped_boxes) = self.det_loader.process(img).read()
            if boxes != None and len(boxes) and show:
                for box in boxes:
                    b1=box.numpy().astype('int32')
                    _img = cv2.rectangle(img,b1[:2],b1[2:],(255,0,0),1)
                    cv2.imshow('box',_img)
            if orig_img is None:
                raise Exception("no image is given")
            if boxes is None or boxes.nelement() == 0:
                pose = self.writer.get_pose(None, None, None, None, None, orig_img)
            else:
                # Pose Estimation
                inps = inps.to(self.args.device)
                if self.args.flip:
                    inps = torch.cat((inps, flip(inps)))
                hm = self.pose_model(inps)
                if self.args.flip:
                    hm_flip = flip_heatmap(hm[int(len(hm) / 2):], self.pose_dataset.joint_pairs, shift=True)
                    hm = (hm[0:int(len(hm) / 2)] + hm_flip) / 2
                hm = hm.cpu()
                pose = self.writer.get_pose(boxes, scores, ids, hm, cropped_boxes, orig_img)
            torch.cuda.empty_cache()
        return pose,inps,boxes,cropped_boxes

    def getImg(self):
        return self.writer.orig_img

    def vis(self, image, pose):
        if pose is not None:
            image = self.writer.vis_frame(image, pose, self.writer.opt, self.writer.vis_thres)
        return image

    def writeJson(self, final_result, outputpath, form='coco', for_eval=False):
        from alphapose.utils.pPose_nms import write_json
        write_json(final_result, outputpath, form=form, for_eval=for_eval)
        print("Results have been written to json.")

