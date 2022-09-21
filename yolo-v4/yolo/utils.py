from .config import *
import cv2
import numpy as np
np.random.seed(SEED)
import colorsys
import random
random.seed(SEED)

class Utils():
    @staticmethod
    def image_preprocess_old(image, target_size, gt_boxes=None):
        ih, iw    = target_size
        h,  w, _  = image.shape

        scale = min(iw/w, ih/h)
        nw, nh  = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
        dw, dh = (iw - nw) // 2, (ih-nh) // 2
        image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
        image_paded = image_paded / 255.

        if gt_boxes is None:
            return image_paded

        else:
            gt_boxes[:, [0, 2]] = ((gt_boxes[:, [0, 2]]*w) * scale + dw)
            gt_boxes[:, [1, 3]] = ((gt_boxes[:, [1, 3]]*h) * scale + dh)
            return image_paded, gt_boxes
    
    def image_preprocess(image, target_size, gt_boxes=None):
        th, tw    = target_size
        h,  w, _  = image.shape

        
        image_resized = cv2.resize(image, (tw, th))
        image_resized = image_resized / 255.
        
        if gt_boxes is None:
            return image_resized

        else:
            return image_resized, gt_boxes
    
    @staticmethod
    def c_xywh2xyxy(box):
        box[:,:2] = box[:,:2] - box[:,2:]*0.5
        box[:,2:] = box[:,2:] + box[:,2:]
        return box
    @staticmethod
    def bboxes_iou(boxes1, boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)
        
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area    = inter_section[..., 0] * inter_section[..., 1]
        union_area    = boxes1_area + boxes2_area - inter_area
        ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

        return ious
    
    @staticmethod
    def draw_bbox(image, bboxes,org_shape, conf=True,scale=False,show_label=True, show_confidence = True, Text_colors=(255,255,0), rectangle_colors='',tracking=False):   
            org_h, org_w = org_shape
            image = cv2.resize(image,(org_w,org_h))
            image_h, image_w, _ = image.shape
            hsv_tuples = [(1.0 * x / NUM_CLASS, 1., 1.) for x in range(NUM_CLASS)]
            #print("hsv_tuples", hsv_tuples)
            colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

            random.seed(0)
            random.shuffle(colors)
            random.seed(None)

            for i, bbox in enumerate(bboxes):
                coor = np.array(bbox[:4], dtype=np.int32)
                if conf:
                    score = bbox[4]
                    class_ind = int(bbox[5])
                else:
                    score = 1.0
                    class_ind = int(bbox[4])
                scl_str = ''
                if scale:
                    scl = int(bbox[6])
                    scl_str = f's:{scl}'
                
                bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
                bbox_thick = int(0.6 * (image_h + image_w) / 1000)
                if bbox_thick < 1: bbox_thick = 1
                fontScale = 0.75 * bbox_thick
                (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

                # put object rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)

                if show_label:
                    # get text label
                    score_str = " {:.2f}".format(score) if show_confidence else ""

                    if tracking: score_str = " "+str(score)

                    #try:
                    label = "{}".format(CLASS_NAMES[class_ind]) + score_str + scl_str
                    #except KeyError:
                    #    print("You received KeyError, this might be that you are trying to use yolo original weights")
                    #    print("while using custom classes, if using custom model in configs.py set YOLO_CUSTOM_WEIGHTS = True")

                    # get text size
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                          fontScale, thickness=bbox_thick)
                    # put filled text rectangle
                    cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)

                    # put text above rectangle
                    cv2.putText(image, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

            return image
    
    @staticmethod
    def random_horizontal_flip(image, bboxes):
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]

        return image, bboxes
    
    @staticmethod
    def random_crop(image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes
    
    @staticmethod
    def random_translate(image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes
    