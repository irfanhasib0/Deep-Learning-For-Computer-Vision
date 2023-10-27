from .utils import Utils
from .config import *
import numpy as np
np.random.seed(SEED)
import cv2
import time
import glob
from copy import deepcopy
from .tf import *
from collections import defaultdict
import json 

with open(f'{DATA_DIR}/COCO/annotations_trainval2017/annotations/instances_val2017.json','r') as file:
    val_data = json.load(file)

class_dict = {val:key for key,val in CLASS_NAMES.items()}
id_dict = {}
for cat in val_data['categories']:
    name = cat['name']
    if name == 'motorcycle' : name = 'motorbike'
    if name == 'airplane' : name = 'aeroplane'
    if name == 'couch' : name = 'sofa'
    if name == 'tv' : name = 'tvmonitor'
    try : id_dict [cat['id']] = class_dict[name]
    except : 
        try : id_dict [cat['id']] = class_dict[name.replace(' ','-')]
        except : id_dict [cat['id']] = class_dict[name.replace(' ','')]
rev_id_dict = {val:key for key,val in id_dict.items()}

class  YoloDecodeNetout():
    def nms(self,bboxes, iou_threshold, sigma=0.3, method='nms'):
        classes_in_img = list(set(bboxes[:, 5]))
        best_bboxes = []

        for cls in classes_in_img:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]
            while len(cls_bboxes) > 0:
                max_ind = np.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                iou = Utils.bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                weight = np.ones((len(iou),), dtype=np.float32)

                assert method in ['nms', 'soft-nms']

                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0

                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))

                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]

        return best_bboxes
    
    def decode_boxes(self,pred_bbox, org_shape, input_size, score_threshold):
        
        valid_scale=[0, np.inf]
        pred_bbox = np.array(pred_bbox)

        pred_xywh = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                    pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
        # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        
        # 3. clip some boxes those are out of range
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:], [input_size - 1, input_size - 1])], axis=-1)
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0

        # 4. discard some invalid boxes
        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

        # 5. discard boxes with low scores
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > score_threshold
        mask = np.logical_and(scale_mask, score_mask)
        coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

        return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
    
    def decode_boxes_org(self,pred_bbox,scale):
        pred_xywh = pred_bbox[:,0:4]
        scores    = pred_bbox[:, 4:5]
        pred_prob = pred_bbox[:, 5:]
        
        pred_coors = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                     pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
        
        classes = np.argmax(pred_prob, axis=-1)

        return np.concatenate([pred_coors, scores, classes[:, np.newaxis],scale], axis=-1)

    
    def detect_video(self,Yolo, video_path, output_path='', input_size=416, show=True, score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):
        times, times_2 = [], []
        vid = cv2.VideoCapture(video_path)

        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))

        if output_path != '': 
            codec = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4

        while True:
            _, img = vid.read()

            try:
                original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                original_image = cv2.resize(original_image, (640, 480))
            except:
                break

            image_data = Utils.image_preprocess(np.copy(original_image), [input_size, input_size])
            image_data = image_data[np.newaxis, ...].astype(np.float32)

            t1 = time.time()
            if YOLO_FRAMEWORK == "tf":
                pred_bbox = Yolo.predict(image_data)
            elif YOLO_FRAMEWORK == "trt":
                batched_input = tf.constant(image_data)
                result = Yolo(batched_input)
                pred_bbox = []
                for key, value in result.items( ):
                    value = value.numpy()
                    pred_bbox.append(value)

            t2 = time.time()
            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox[:2]]
            pred_bbox = tf.concat(pred_bbox, axis=0)

            bboxes = self.decode_boxes(pred_bbox, original_image, input_size, score_threshold)
            bboxes = self.nms(bboxes, iou_threshold, method='nms')

            image = Utils.draw_bbox(original_image, bboxes, rectangle_colors=rectangle_colors)

            t3 = time.time()
            times.append(t2-t1)
            times_2.append(t3-t1)

            times = times[-20:]
            times_2 = times_2[-20:]

            ms = sum(times)/len(times)*1000
            fps = 1000 / ms
            fps2 = 1000 / (sum(times_2)/len(times_2)*1000)

            image = cv2.putText(image, "Time: {:.1f}FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            
            print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
            if output_path != '': out.write(image)
            if show:
                cv2.imshow('output', image)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break

        cv2.destroyAllWindows()
        
    def detect_images(self,Yolo, img_path, output_path='', input_size=416, show=True, score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):
        times, times_2 = [], []
        
        # by default VideoCapture returns float instead of int
        for i,path in enumerate(glob.glob(img_path+'/*')):
            img = cv2.imread(path)
            width  = img.shape[0]
            height = img.shape[1]
            
            original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_image = cv2.resize(original_image, (640, 480))
            
            image_data = Utils.image_preprocess(np.copy(original_image), [input_size, input_size])
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            
            t1 = time.time()
            if YOLO_FRAMEWORK == "tf":
                #temp = deepcopy(Yolo)
                pred_bbox = Yolo.predict(image_data)
                #Yolo = temp
            elif YOLO_FRAMEWORK == "trt":
                batched_input = tf.constant(image_data)
                result = Yolo(batched_input)
                pred_bbox = []
                for key, value in result.items():
                    value = value.numpy()
                    pred_bbox.append(value)
            
            if i==5:break
        
            t2 = time.time()
            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox[:2]]
            pred_bbox = tf.concat(pred_bbox, axis=0)

            bboxes = self.decode_boxes(pred_bbox, original_image, input_size, score_threshold)
            bboxes = self.nms(bboxes, iou_threshold, method='nms')
            image = Utils.draw_bbox(original_image, bboxes, rectangle_colors=rectangle_colors)

            t3 = time.time()
            times.append(t2-t1)
            times_2.append(t3-t1)

            times = times[-20:]
            times_2 = times_2[-20:]

            ms = sum(times)/len(times)*1000
            fps = 1000 / ms
            fps2 = 1000 / (sum(times_2)/len(times_2)*1000)
            
            image = cv2.putText(image, "Time: {:.1f}FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            
            print("Time : {:.2f}ms, FPS: {:.1f}, Avg. FPS: {:.1f}".format(ms, fps, fps2))
            #print(output_path+path)
            if output_path != '': 
                path = output_path+path.split('/')[-1]
                print(path)
                cv2.imwrite(path,image)
            
            if show:
                cv2.imshow('output', image)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break
            
        cv2.destroyAllWindows()
        
    def detect_image(self,Yolo,decoder, input_img,root='', output_path='', input_size=416, show=True, score_threshold=0.3, iou_threshold=0.45, rectangle_colors='',draw=True):
        
        times, times_2 = [], []
        
        img = cv2.imread(root+input_img['file_name'])
        width  = img.shape[0]
        height = img.shape[1]

        #original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image = cv2.resize(img, (input_size, input_size))

        image_data = Utils.image_preprocess(np.copy(original_image), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        
        t1 = time.time()
        if YOLO_FRAMEWORK == "tf":
            pred_bbox = Yolo.predict(image_data)
            pred_bbox = decoder(pred_bbox,1)
        elif YOLO_FRAMEWORK == "trt":
            batched_input = tf.constant(image_data)
            result = Yolo(batched_input)
            pred_bbox = []
            for key, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)

        t2 = time.time()
        
        pred_seg  = pred_bbox[3:]
        #import pdb;pdb.set_trace()
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox[:3]]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        org_shp = [input_img['height'],input_img['width']]
        bboxes = self.decode_boxes(pred_bbox,org_shp , input_size, score_threshold)
        bboxes = self.nms(bboxes, iou_threshold, method='nms')
        
        results = []
        nbboxes = []
        for ind,bbox in enumerate(bboxes):
            _bbox = bbox.copy()
            hfact = input_img['height']#/input_size
            wfact = input_img['width']#/input_size
            #hfact = wfact = 1/input_size
            _bbox[0] = _bbox[0] * wfact
            _bbox[1] = _bbox[1] * hfact
            _bbox[2] = _bbox[2] * wfact
            _bbox[3] = _bbox[3] * hfact
            #_nbbox = [_bbox[0],_bbox[1], _bbox[2]-_bbox[0], _bbox[3]-_bbox[1]]
            #_mbboxes = [_bbox[0],_bbox[1], _bbox[2], _bbox[3]]
            results+=[{'image_id' : input_img['id'],
                       'file_name':input_img['file_name'],
                       'height': input_img['height'],
                       'width' : input_img['width'],
                       'bbox'  : [_bbox[0],_bbox[1], _bbox[2]-_bbox[0], _bbox[3]-_bbox[1]],
                       'score' : _bbox[4],
                       'category_id' : rev_id_dict[int(_bbox[5])]}]
            bboxes[ind][:4] = [_bbox[0],_bbox[1], _bbox[2], _bbox[3]]
        if not draw :
            return results
        image = Utils.draw_bbox(original_image, bboxes, org_shp, rectangle_colors=rectangle_colors)

        t3 = time.time()
        times.append(t2-t1)
        times_2.append(t3-t1)

        times = times[-20:]
        times_2 = times_2[-20:]

        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        fps2 = 1000 / (sum(times_2)/len(times_2)*1000)

        image = cv2.putText(image, "Time: {:.1f}FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        
        print("Time: {:.2f}ms,FPS: {:.1f}, Avg. FPS: {:.1f}".format(ms, fps, fps2))
        return image,pred_seg,results
        
        
        