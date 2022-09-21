from .utils import Utils
from .config import *
from .tf import *

from tqdm import trange
from collections import defaultdict
import os
import shutil
import cv2
import numpy as np
np.random.seed(SEED)
import time


def voc_ap(rec, prec):
    
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre

def bboxes_iou(boxes1, boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = (boxes1[2] - boxes1[0]) * (boxes1[3] - boxes1[1])
        boxes2_area = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

        left_up       = np.maximum(boxes1[:2], boxes2[:2])
        right_down    = np.minimum(boxes1[2:], boxes2[2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area    = inter_section[0] * inter_section[1]
        union_area    = boxes1_area + boxes2_area - inter_area
        ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

        return ious
    

def get_mAP(Yolo, dataset, decoder, min_overlap = 0.5, score_threshold=0.25, iou_threshold=0.50, TEST_INPUT_SIZE=TEST_INPUT_SIZE):
    ground_truth_dir_path = 'mAP/ground-truth'
    if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
    if not os.path.exists(ground_truth_dir_path): os.makedirs(ground_truth_dir_path)
        
    img_dir_path = 'mAP/img'
    if os.path.exists(img_dir_path): shutil.rmtree(img_dir_path)
    if not os.path.exists(img_dir_path): os.makedirs(img_dir_path)
        
    print(f'\ncalculating mAP{int(iou_threshold*100)}...\n')

    gt_counter_per_class = defaultdict(int)
    gt_dict=defaultdict(list)
    img_path_dict={}
    for index in trange(dataset.num_samples):
        ann_dataset = dataset.annotations[index]
        img_path, bbox_data_gt = dataset.parse_annotation(ann_dataset, True)
        
        bboxes_gt = []; classes_gt = []
        if len(bbox_data_gt): 
            bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
            
        #bounding_boxes = []
        for i in range(len(bboxes_gt)):
            gt_dict[index].append({"class_name":CLASS_NAMES[classes_gt[i]], "bbox":bboxes_gt[i], "used":False})
            gt_counter_per_class[CLASS_NAMES[classes_gt[i]]] += 1
            '''
            print(CLASS_NAMES[classes_gt[i]])
            x1,y1,x2,y2=int(bboxes_gt[i][0]*TEST_INPUT_SIZE),int(bboxes_gt[i][1]*TEST_INPUT_SIZE),int(bboxes_gt[i][2]*TEST_INPUT_SIZE),int(bboxes_gt[i][3]*TEST_INPUT_SIZE)
            gt_img=cv2.imread(img_path)
            gt_img=cv2.resize(gt_img,(TEST_INPUT_SIZE,TEST_INPUT_SIZE))
            cv2.rectangle(gt_img, (x1, y1), (x2,y2), (255,0,0))
            plt.imshow(gt_img)
            plt.show()
            '''
        img_path_dict[index]=img_path
        
        #gt_dict[index].append(bounding_boxes)
    #print(gt_counter_per_class)
    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)
    n_classes  = len(gt_classes)
    
    times = []
    json_pred = defaultdict(list)
    for index in trange(dataset.num_samples):
        ann_dataset = dataset.annotations[index]

        original_image = cv2.imread(img_path_dict[index])
        image = Utils.image_preprocess(original_image, [TEST_INPUT_SIZE, TEST_INPUT_SIZE])
        image_data = image[np.newaxis, ...].astype(np.float32)
            
        t1 = time.time()
        pred_bbox = Yolo.predict(image_data)
        t2 = time.time()
        times.append(t2-t1)
        
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox[:2] ]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        
        bboxes = decoder.decode_boxes(pred_bbox, original_image, TEST_INPUT_SIZE, score_threshold)
        bboxes = decoder.nms(bboxes, iou_threshold, method='nms')
        image  = Utils.draw_bbox(original_image, bboxes, rectangle_colors='')
        #plt.imshow(image)
        #plt.show()
        #cv2.imwrite(img_dir_path+'/'+str(index)+'.png',image)
        for bbox in bboxes:
            coor = np.array(bbox[:4], dtype=np.float32)/TEST_INPUT_SIZE
            score = bbox[4]
            class_ind = int(bbox[5])
            class_name = CLASS_NAMES[class_ind]
            score = '%.4f' % score
            
            json_pred[class_name].append({"confidence": score, "file_id": index, "bbox": coor})
                
    ms = sum(times)/len(times)*1000
    fps = 1000 / ms
    
    # Calculate the AP for each class
    sum_AP = 0.0
    ap_dictionary = {}
    # open file to store the results
    with open("mAP/results.txt", 'w') as results_file:
        results_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}
        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            # Load predictions of that class
            predictions_data =  json_pred[class_name] 
            predictions_data.sort(key=lambda x: x['confidence'], reverse=True)
            
            nd = len(predictions_data)
            tp = [0] * nd # creates an array of zeros of size nd
            fp = [0] * nd
            for idx, prediction in enumerate(predictions_data):
                file_id = prediction["file_id"]
                bb = [x for x in prediction["bbox"] ] 
                
                ground_truth_data = gt_dict[file_id]
                ovmax = -1
                for obj_idx,obj in enumerate(ground_truth_data):
                    if obj["class_name"] == class_name:
                        bbgt = [x for x in obj["bbox"]] # bounding box of ground truth
                        bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                        iw = max(bi[2] - bi[0] +1,0.0)
                        ih = max(bi[3] - bi[1] +1,0.0)
                        
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0]+1) * (bb[3] - bb[1]+1) + (bbgt[2] - bbgt[0]+1) * (bbgt[3] - bbgt[1]+1) - (iw * ih)
                        if ua>0: ov = (iw * ih) / ua
                        else : ov=0
                        #ov = bboxes_iou(bbgt,bb)
                        if ov > ovmax:
                            ovmax = ov
                            max_idx=obj_idx
                #print(class_name,ovmax)
                if ovmax >= min_overlap:# if ovmax > minimum overlap
                    if ground_truth_data[max_idx]["used"] == False:
                        # true positive
                        tp[idx] = 1
                        ground_truth_data[max_idx]["used"] = True
                        count_true_positives[class_name] += 1
                        
                    else: fp[idx] = 1
                else:   fp[idx] = 1
            
            
            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            #print('FP :', fp)
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val
            #print('TP :', tp)
            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
            #print('Rec :', rec)
            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
            #print('Prec :', prec)

            ap, mrec, mprec = voc_ap(rec, prec)
            sum_AP += ap
            text = "{0:.3f}%".format(ap*100) + " = " + class_name + " AP  " #class_name + " AP = {0:.2f}%".format(ap*100)

            rounded_prec = [ '%.3f' % elem for elem in prec ]
            rounded_rec = [ '%.3f' % elem for elem in rec ]
            # Write to results.txt
            results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")

            print(text)
            ap_dictionary[class_name] = ap

        results_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / n_classes

        text = "mAP = {:.3f}%, {:.2f} FPS".format(mAP*100, fps)
        results_file.write(text + "\n")
        print(text)
        
        return mAP*100
    
    
def calc_mAP(Yolo, dataset, decoder, min_overlap = 0.5, score_threshold=0.25, iou_threshold=0.50, TEST_INPUT_SIZE=TEST_INPUT_SIZE):   
    print(f'\ncalculating mAP{int(iou_threshold*100)}...\n')
    _gts = defaultdict(list)       # gt for evaluation
    _dts = defaultdict(list)       # dt for evaluation
    for _id,gt in enumerate(gts):
        gt['id'] = _id
        gt['_ignore']=0
        _gts[gt['image_id'], gt['category_id']].append(gt)
    for _id,dt in enumerate(dts):
        dt['id'] = _id
        _dts[dt['image_id'], dt['category_id']].append(dt)

    gt_counter_per_class = defaultdict(int)
    gt_dict=defaultdict(list)
    img_path_dict={}
    for index in trange(dataset.num_samples):
        ann_dataset = dataset.annotations[index]
        img_path, bbox_data_gt = dataset.parse_annotation(ann_dataset, True)
        
        bboxes_gt = []; classes_gt = []
        if len(bbox_data_gt): 
            bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
            
        for i in range(len(bboxes_gt)):
            gt_dict[index].append({"class_name":CLASS_NAMES[classes_gt[i]], "bbox":bboxes_gt[i], "used":False})
            gt_counter_per_class[CLASS_NAMES[classes_gt[i]]] += 1
        img_path_dict[index]=img_path
        
    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)
    n_classes  = len(gt_classes)
    
    times = []
    json_pred = defaultdict(list)
    for index in trange(dataset.num_samples):
        ann_dataset = dataset.annotations[index]

        original_image = cv2.imread(img_path_dict[index])
        image = Utils.image_preprocess(original_image, [TEST_INPUT_SIZE, TEST_INPUT_SIZE])
        image_data = image[np.newaxis, ...].astype(np.float32)
            
        t1 = time.time()
        pred_bbox = Yolo.predict(image_data)
        t2 = time.time()
        times.append(t2-t1)
        
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox[:2] ]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        
        bboxes = decoder.decode_boxes(pred_bbox, original_image, TEST_INPUT_SIZE, score_threshold)
        bboxes = decoder.nms(bboxes, iou_threshold, method='nms')
        image  = Utils.draw_bbox(original_image, bboxes, rectangle_colors='')
        for bbox in bboxes:
            coor = np.array(bbox[:4], dtype=np.float32)/TEST_INPUT_SIZE
            score = bbox[4]
            class_ind = int(bbox[5])
            class_name = CLASS_NAMES[class_ind]
            score = '%.4f' % score
            
            json_pred[class_name].append({"confidence": score, "file_id": index, "bbox": coor})
                
    ms = sum(times)/len(times)*1000
    fps = 1000 / ms
    
    # Calculate the AP for each class
    sum_AP = 0.0
    ap_dictionary = {}
    # open file to store the results
    with open("mAP/results.txt", 'w') as results_file:
        results_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}
        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            # Load predictions of that class
            predictions_data =  json_pred[class_name] 
            predictions_data.sort(key=lambda x: x['confidence'], reverse=True)
            
            nd = len(predictions_data)
            tp = [0] * nd # creates an array of zeros of size nd
            fp = [0] * nd
            for idx, prediction in enumerate(predictions_data):
                file_id = prediction["file_id"]
                bb = [x for x in prediction["bbox"] ] 
                
                ground_truth_data = gt_dict[file_id]
                ovmax = -1
                for obj_idx,obj in enumerate(ground_truth_data):
                    if obj["class_name"] == class_name:
                        bbgt = [x for x in obj["bbox"]] # bounding box of ground truth
                        bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                        iw = max(bi[2] - bi[0] +1,0.0)
                        ih = max(bi[3] - bi[1] +1,0.0)
                        
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0]+1) * (bb[3] - bb[1]+1) + (bbgt[2] - bbgt[0]+1) * (bbgt[3] - bbgt[1]+1) - (iw * ih)
                        if ua>0: ov = (iw * ih) / ua
                        else : ov=0
                        #ov = bboxes_iou(bbgt,bb)
                        if ov > ovmax:
                            ovmax = ov
                            max_idx=obj_idx
                #print(class_name,ovmax)
                if ovmax >= min_overlap:# if ovmax > minimum overlap
                    if ground_truth_data[max_idx]["used"] == False:
                        # true positive
                        tp[idx] = 1
                        ground_truth_data[max_idx]["used"] = True
                        count_true_positives[class_name] += 1
                        
                    else: fp[idx] = 1
                else:   fp[idx] = 1
            
            
            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            #print('FP :', fp)
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val
            #print('TP :', tp)
            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
            #print('Rec :', rec)
            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
            #print('Prec :', prec)

            ap, mrec, mprec = voc_ap(rec, prec)
            sum_AP += ap
            text = "{0:.3f}%".format(ap*100) + " = " + class_name + " AP  " #class_name + " AP = {0:.2f}%".format(ap*100)

            rounded_prec = [ '%.3f' % elem for elem in prec ]
            rounded_rec = [ '%.3f' % elem for elem in rec ]
            # Write to results.txt
            results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")

            print(text)
            ap_dictionary[class_name] = ap

        results_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / n_classes

        text = "mAP = {:.3f}%, {:.2f} FPS".format(mAP*100, fps)
        results_file.write(text + "\n")
        print(text)
        
        return mAP*100