#!/usr/bin/env python
# coding: utf-8

# In[1]:


#29/04/2022
#================================================================
# Yolo V-3 by irfanhasib.me@gmail.com
# Inspired by -
# GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#================================================================
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED']=str(0)
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import time
import glob
import pandas as pd
from matplotlib import pyplot as plt
from multiprocessing import Process, Queue, Pipe
import time
import shutil
import json
from tqdm import tqdm,trange
import pickle
import zlib
from datetime import datetime


# In[2]:


from yolo.model import YoloModel, calc_yolo_loss, calc_seg_loss
from yolo.decoder import YoloDecodeNetout
from yolo.dataset import Dataset
from yolo.eval import get_mAP
from yolo.utils import Utils
from yolo.seg_loader import Seg_Utils
from yolo.config import *
from yolo.tf import *


# In[3]:


yolo_test = False
yolo_eval = False
seg_test  = False
sanity_check = False
data_gen  = DATA_GEN
debug     = False


# In[4]:


if yolo_test == True or yolo_eval== True or seg_test==True or sanity_check== True or debug == True:
    save_notebook = False
else:
    save_notebook = True
    
if save_notebook == True:
    if not os.path.exists(TRAIN_CHECKPOINTS_FOLDER): os.makedirs(TRAIN_CHECKPOINTS_FOLDER)
    with open(TRAIN_CHECKPOINTS_FOLDER +'/params.txt','w') as file:
        log_str='Time '.ljust(30)+': '+str(datetime.now())+'\n'
        for key in list(params.keys())[:-1]:
            if key[:2] != '__':
                log_str += key.ljust(30)+': ' + str(params[key])+'\n'
        print(log_str)
        file.write(log_str)


# In[5]:


if save_notebook == True:
    curr_time=time.time()
    print('System time : ',curr_time)
    #%autosave 1
    #time.sleep(3)
    
    os.system(f"cp yolo-v3.ipynb {TRAIN_CHECKPOINTS_FOLDER}/yolo-v3_{str(curr_time)}.ipynb")
    if not os.path.exists(TRAIN_CHECKPOINTS_FOLDER+'/yolo/'): os.makedirs(TRAIN_CHECKPOINTS_FOLDER+'/yolo/')
    files = glob.glob('yolo/*')
    for file in files:
        try : os.system(f"cp -r {file} {TRAIN_CHECKPOINTS_FOLDER}/{file}")
        except PermissionError:
            print('PermissionError : ',file)
    #%autosave 120


# In[6]:


os.system("ls")


# In[7]:


#yolo = YoloModel(training=True,N=1)
#yolo_model=yolo.get_model()
#yolo_model.summary()


# In[8]:


if yolo_test == True:
    if not os.path.exists(TRAIN_CHECKPOINTS_FOLDER+'/pred_imgs'): os.makedirs(TRAIN_CHECKPOINTS_FOLDER+'/pred_imgs')
    #video_path   = "./IMAGES/test.mp4"
    img_path   = "/home/irfan/Desktop/Code/Datasets/COCO/val2017/"
    yolo = YoloModel()
    yolo_model=yolo.get_model()
    decoder = YoloDecodeNetout()
    for i,layer in enumerate(yolo_model.layers):
        yolo_model.get_layer(layer.name).trainable=False
    #decoder.detect_video(yolo_model, video_path, input_size=288, show=True, score_threshold=0.1, iou_threshold=0.2, rectangle_colors='')
    decoder.detect_images(yolo_model, img_path, output_path=TRAIN_CHECKPOINTS_FOLDER+'/pred_imgs/',input_size=256, show=True, score_threshold=0.3, iou_threshold=0.5, rectangle_colors='')


# In[9]:


#!ls /home/irfan/Desktop/Code/Datasets/COCO/val2017/


# In[10]:


if yolo_eval == True:
    res_dict=[]
    for min_overlap in list(range(50,100,5)):
        min_overlap = min_overlap/100
        for iou_threshold in [0.1]:#,0.2,0.3,0.3,0.4,0.5]:#[0.1,0.2,0.3,0.4,0.5]:
            for score_threshold in [0.0]:#,0.05,0.1,0.2]:#[0.1,0.2,0.3,0.4,0.5]:
                yolo = YoloModel()
                yolo_model=yolo.get_model()
                decoder = YoloDecodeNetout()

                testset = Dataset('test')
                out=get_mAP(yolo_model, testset, decoder, min_overlap= min_overlap ,score_threshold=score_threshold, iou_threshold=iou_threshold, TEST_INPUT_SIZE=YOLO_INPUT_SIZE)
                res_dict+=[[out,min_overlap,score_threshold,iou_threshold]]
                print(res_dict)
    
    with open(TRAIN_CHECKPOINTS_FOLDER+'/scores_0.1_0.0001.pkl','wb') as file:
        pickle.dump(res_dict,file)
    print(sum([res[0] for res in res_dict])/10)


# In[11]:


if seg_test == True:
    trainset = Dataset('train')
    testset = Dataset('test')
    
    yolo = YoloModel()
    yolo_model=yolo.get_model()
        
    for image , label in trainset:
        break
        
    out = yolo_model.predict(image)
    plt.imshow(label[3][1].max(axis=-1))
    plt.show()
    plt.imshow(out[2][1][:,:,:len(CLASS_NAMES)].max(axis=-1))
    plt.show()

    for image , label in testset:
        break
    out = yolo_model.predict(image)
    plt.imshow(label[3][1].max(axis=-1))
    plt.show()
    plt.imshow(out[2][1][:,:,:len(CLASS_NAMES)].max(axis=-1))
    plt.show()
    
    #plt.imshow(label[3][1])
    #plt.show()
    #plt.imshow(out[2][0][:,:].max(axis=-1))
    #plt.imshow(image[0])
    #plt.show()


# In[12]:


if sanity_check == True:
    trainset = Dataset('train')
    testset  = Dataset('test')
    for train_img, train_label in trainset:
        print('..')
        break
    for test_img, test_label in testset:
        print('...')
        break
    decoder = YoloDecodeNetout()
    pred_bbox = [label[0][0] for label in train_label]#[label_sbbox, label_mbbox, label_lbbox]
    
    pred_bbox = [np.reshape(x, (-1, np.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = np.concatenate(pred_bbox, axis=0)

    bboxes = decoder.decode_boxes(pred_bbox, train_img[0], YOLO_INPUT_SIZE, TEST_SCORE_THRESHOLD)
    bboxes = decoder.nms(bboxes, TEST_IOU_THRESHOLD, method='nms')

    out=Utils.draw_bbox(train_img[0], bboxes, conf=True,show_label=True, show_confidence = True, Text_colors=(255,255,0), rectangle_colors='', tracking=False)
    plt.imshow(train_img[0])
    plt.show()
    plt.imshow(out)


# In[13]:


wts_check=False
if wts_check== True:
    wts = yolo_model.trainable_weights
    for i in range(len(wts)):
        if len(wts[i].shape) and wts[i].shape[0]==3: 
            print(wts[i].shape)
            _wts = tf.abs(wts[i])
            vector = tf.reduce_sum(_wts,axis=[0,1])
            norm_rev_vec = 1 - vector/tf.reduce_max(vector)
            plt.imshow(norm_rev_vec,cmap='gray')
            print(tf.reduce_mean(norm_rev_vec))
            plt.show()


# In[14]:


class loss_dict_obj(dict):
        def tf2np(self,val):
            if hasattr(val,'numpy'):
                val=val.numpy()
            else:
                if val==None: val=0
            return val

        def sum_update(self,c_dict):
            for key,val in c_dict.items():
                if key in list(self.keys()):
                    self[key]+=self.tf2np(c_dict[key])
                else:
                    self[key]=self.tf2np(c_dict[key])


        def ext_update(self,c_dict,_ext='_ext'):
            for key,val in c_dict.items():
                     self[_ext+key]=self.tf2np(val)

        def divide(self,div_val):
            for key,val in self.items():
                if type(div_val)==dict or type(div_val)==loss_dict_obj:
                    self[key]/=div_val[key]
                else:
                    self[key]/=div_val

        def _sum(self):
            total=0
            for val in self.values():
                total+=val
            return total

        def copy_keys(self,c_dict,keys):
            for key in keys:
                self[key]=c_dict[key]

        def apply(self,func):
            for key in self.keys():
                self[key]=self.tf2np(func(self[key]))

def get_best_model_path(exp_dir = 'logs/exp-D101'):
    paths = glob.glob(f'{exp_dir}/model/epoch_*')
    arg = np.argmin([float(path.split('_')[-1]) for path in paths])
    best_model_path = paths[arg] + '/weights'
    print("Found best model path : ",best_model_path)
    return best_model_path


def save_loss_logs(loss_dict,epoch):
    if epoch==0: log_str=','.join(list(loss_dict.keys()))+'\n'
    else : log_str = ''
    log_str += ','.join(list(map(str,loss_dict.values())))+'\n'

    with open(os.path.join(TRAIN_CHECKPOINTS_FOLDER,'loss.csv'),'a+') as file:
        file.write(log_str)
            
def save_std_logs(train_loss,all_logs,epoch):
    if epoch==0 : log_str='epoch,'+','.join(list(train_loss.keys()))+'\n'
    else : log_str = ''
            
    for _ind in range(no_train_batch): 
        log_str += str(epoch)+','
        log_str += ','.join(list(map(str,all_logs[_ind])))+'\n'

    with open(os.path.join(TRAIN_CHECKPOINTS_FOLDER,'all_loss.csv'),'a+') as file:
        file.write(log_str)
        
def save_sample_losses(sample_losses,epoch):
    
    for _lind,loss_name in zip([0,1],['det','seg']):
        if TRAIN_LOSS_WTS[_lind]:
            if epoch==0 : log_str='epoch,'+','.join(list(map(str,range(TRAIN_BATCH_SIZE))))+'\n'
            else : log_str = ''
            for _ind in range(no_train_batch): 
                log_str += str(epoch)+','
                log_str += ','.join(list(map(str,sample_losses[_ind][_lind].numpy())))+'\n'

            with open(os.path.join(TRAIN_CHECKPOINTS_FOLDER,'sample_loss_'+loss_name+'.csv'),'a+') as file:
                file.write(log_str)  


@tf.function
def train_step(image_data, target,epoch,alpha=1.0):
        #tf.reset_dafault_graph()
        train_loss_dict = loss_dict_obj()
        sample_loss_dict= loss_dict_obj()
        sample_seg_losses = sample_det_losses = det_loss = seg_loss = kl_coef = 0.0
        giou_loss = conf_loss = prob_loss = 0.0
        gradients1 = gradients2 = [None]*len(yolo_model.trainable_variables)
        grad_variance1 = grad_variance2 = []
        smp_grads_det = [] ; smp_grads_seg = []
        yolo_model.training = True
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([yolo_model.trainable_variables])
            pred_result = yolo_model(image_data)
            pred_result = yolo.decode_output(pred_result)
            del image_data
            
            if TRAIN_LOSS_WTS[0]:
                for i in range(NO_OF_GRID):
                    conv, pred = pred_result[i*2], pred_result[i*2+1]
                    #tf.print(pred.shape,conv.shape,target[i][0].shape)
                    loss_dict, _giou_loss , _conf_loss , _prob_loss = calc_yolo_loss(pred, conv, *target[i], i)
                    train_loss_dict.sum_update(loss_dict)
                    
                    giou_loss += 1/NO_OF_GRID * _giou_loss
                    conf_loss += 1/NO_OF_GRID * _conf_loss
                    prob_loss += 1/NO_OF_GRID * _prob_loss
                    
                loss =  (MTL_LOSS_WTS[0] * giou_loss + MTL_LOSS_WTS[1] * conf_loss)
                grads_1 = tape.gradient(loss , yolo_model.trainable_variables)
                grads_2 = tape.gradient(MTL_LOSS_WTS[2] * prob_loss , yolo_model.trainable_variables)
                
        return train_loss_dict, [grads_1,grads_2]

def reduce_prob(x):
    mean = tf.math.reduce_mean(x,axis=0)
    #std  = tf.math.reduce_std(x,axis=0)
    var  = tf.reduce_mean(tf.square(x - mean),axis=0) + 1e-20
    p_x  = tf.square((x - mean)) / var
    p_x  = tf.exp(-0.5*p_x)
    coef = 1/(tf.sqrt(2*np.pi)*var)
    p_x  = coef * p_x
    return (p_x/tf.reduce_sum(p_x,axis=0)) + 1e-20

def reduce_kl_div(x,y):
    p_x = reduce_prob(x)
    p_y = reduce_prob(y)
    kl_div = p_x * tf.math.log(p_x / p_y)
    kl_div = tf.reduce_sum(kl_div,axis=0)
    kl_div = tf.clip_by_value(kl_div,0,1)
    return kl_div

def get_probs(_grads,_grads_conf_1):
    coef_1 = 1/_vars
    coef_2 = tf.square(_grads - _means)/_vars
    probs  = coef_1 * tf.exp(-0.5*coef_2)
    probs  = (probs / tf.reduce_max(probs)) + 1e-10
    return probs

def get_disc_probs(_grads,_confs):
    sign      = np.sign(_confs)
    pos_prob  = np.mean((sign + 1.0)/2.0,axis=0)
    neg_prob  = np.mean(np.abs(sign - 1.0)/2.0,axis=0)
    pos_mask  = np.uint8(_grads>=0)
    neg_mask  = np.uint8(_grads<0)
    
    _grads    = pos_mask * pos_prob + neg_mask * neg_prob
    return _grads

def reduce_disc_kl_div(p_x,p_y):
    kl_div = p_x * tf.math.log(p_x / p_y)
    kl_div = tf.reduce_sum(kl_div,axis=0)
    kl_div = tf.clip_by_value(kl_div,0,1)
    return kl_div

#@tf.function    
def train_batch(image_data,target,frozen=False):
    global model_flag, model_flag_aux
    alpha = MTL_USE_ALPHA
    gamma = MTL_LR_BIAS
    NO_MINI_BATCH = TRAIN_BATCH_SIZE//TRAIN_MINI_BATCH_SIZE
    gradients1 = [0.0]*len(yolo_model.trainable_variables)
    gradients2 = [0.0]*len(yolo_model.trainable_variables)
    loss_wts_grads = [0.0]* 4 #MTL_NO_OF_BLOCKS
    norms_matrix   = [0.0]* 4
    gradient_vars1 = [0.0] * MTL_NO_OF_SHARED_LAYERS
    gradient_vars2 = [0.0] * MTL_NO_OF_SHARED_LAYERS
    train_loss = loss_dict_obj(); smp_loss = loss_dict_obj()
    
    smp_grads_det = []
    smp_grads_seg = []
    for ind in range(0,NO_MINI_BATCH):
        mbatch_image,mbatch_target = get_mini_batch(image_data,target,ind,TRAIN_MINI_BATCH_SIZE)
        mbatch_train_loss, mbatch_grads = train_step(mbatch_image, mbatch_target,epoch)
        
        _coef = 1/NO_MINI_BATCH
        for ind,[lgrad1,lgrad2] in enumerate(zip(*mbatch_grads)):           
            if type(lgrad1) == type(None) : lgrad1 = 0.0
            if type(lgrad2) == type(None) : lgrad2 = 0.0
            gradients1[ind] += _coef * lgrad1
            gradients2[ind] += _coef * lgrad2
                   
        train_loss.sum_update(mbatch_train_loss)
    train_loss.divide(NO_MINI_BATCH)    
    if not frozen:
        #st_offset = NO_FROZEN_LAYERS
        st_offset = 0
        for ind in range(st_offset,len(grads_conf_1)):
            grads_conf_1[ind][0:MTL_GRADS_QLEN-1] = grads_conf_1[ind][1:MTL_GRADS_QLEN]
            grads_conf_1[ind][MTL_GRADS_QLEN-1]   = gradients1[ind-st_offset]

            grads_conf_2[ind][0:MTL_GRADS_QLEN-1] = grads_conf_2[ind][1:MTL_GRADS_QLEN]
            grads_conf_2[ind][MTL_GRADS_QLEN-1]   = gradients2[ind-st_offset]

    kl_divs = []
    n_layers = len(gradients1)
    for ind in range(n_layers):
        beta = ind/n_layers
        if not frozen:
            p_x = get_disc_probs(gradients1[ind],grads_conf_1[st_offset+ind])
            p_y = get_disc_probs(gradients2[ind],grads_conf_2[st_offset+ind])
            #kl_div = reduce_disc_kl_div(p_x,p_y)
            
            if MTL_USE_SIG_PROB:
                p_x = tf.nn.sigmoid(10.0*p_x - 5.0) # From 0-1 to -5 to 5 to 0-1
                p_y = tf.nn.sigmoid(10.0*p_y - 5.0)
                p_xy = (p_x * p_y) + 10e-20
                p_xy = tf.math.sqrt(p_xy) - 0.5
                p_x  = p_x - 0.5
                p_y  = p_y - 0.5
                
            if MTL_USE_MUL_PROB: 
                gradients1[ind] = (gamma + alpha*p_xy) * (gradients1[ind] + gradients2[ind])
                
            #elif MTL_USE_IND_PROB:
            #    gradients1[ind] = (gamma + alpha*p_x) * gradients1[ind] + (gamma + alpha*p_y) * gradients2[ind]
            elif MTL_USE_IND_PROB:
                gradients1[ind] = (gamma + alpha*p_x) * gradients1[ind] + (gamma + alpha*p_y) * gradients2[ind]
           
            elif MTL_USE_COMB_PROB:
                grad_1 = (gamma + alpha*p_x*p_y) * (gradients1[ind] + gradients2[ind])
                grad_2 = (gamma + alpha*p_x) * gradients1[ind] + (gamma + alpha*p_y) * gradients2[ind]
                gradients1[ind] = (1-beta) * grad_1 + beta * grad_2
            else : 
                gradients1[ind] = gradients1[ind] + gradients2[ind]            
            #kl_divs += [kl_div]
        else:
            gradients1[ind] = gradients1[ind] + gradients2[ind]
        
    
    optimizer.apply_gradients(zip(gradients1, yolo_model.trainable_variables))
        
    return train_loss,smp_grads_det,smp_grads_seg,norms_matrix

def add_ext(loss_dict,_ext='val_'):
    _loss_dict = {}
    for key in loss_dict.keys():
          _loss_dict[_ext+key]=loss_dict[key]
    return _loss_dict

#@tf.function
def validate_step(image_data, target):
    val_loss_dict = loss_dict_obj()
    
    pred_result = yolo_model(image_data)
    pred_result = yolo.decode_output(pred_result,batch_size=TEST_BATCH_SIZE)
    del image_data

    for i in range(NO_OF_GRID):
        conv, pred = pred_result[i*2], pred_result[i*2+1]
        loss_dict, _, _,_ = calc_yolo_loss(pred, conv, *target[i], i,batch_size=TEST_BATCH_SIZE)
        loss_dict = add_ext(loss_dict,_ext='val_')
        val_loss_dict.sum_update(loss_dict)

    if TRAIN_USE_DST or TRAIN_USE_SEG:
        seg_pred     = pred_result[2*(grid -1)+2:]
        seg_label    = target[3:]

        loss_dict, _, _,_ = calc_seg_loss(seg_label,seg_pred)
        loss_dict = add_ext(loss_dict,_ext='val_')
        val_loss_dict.sum_update(loss_dict)

    del pred_result, target
        
    return val_loss_dict


# In[15]:


if len(TRAIN_LRES_PRE_WTS):
    yolo = YoloModel(training=True,input_size=TRAIN_HRES_INPUT_SIZE)
    yolo_model = yolo.get_model()
    model_path = get_best_model_path(exp_dir=TRAIN_LRES_PRE_WTS)
    yolo_model.load_weights(model_path)
    
    trainset = Dataset('train',input_size=TRAIN_HRES_INPUT_SIZE)
    testset  = Dataset('test',input_size=TRAIN_HRES_INPUT_SIZE)

else:
    yolo = YoloModel(training=True)
    yolo_model = yolo.get_model()
    trainset = Dataset('train')
    testset  = Dataset('test')

steps_per_epoch = len(trainset)

optimizer  = tf.keras.optimizers.Adam(lr = TRAIN_LR)
#loss_opt   =  tf.keras.optimizers.Adam(lr = TRAIN_LOSS_WTS_LR)
best_val_loss = 10e8 # should be large at start
no_train_batch = trainset.num_batchs
no_val_batch  = testset.num_batchs
yolo_model.summary()
MTL_NO_OF_SHARED_LAYERS = len(yolo_model.trainable_variables)
for i,layer in enumerate(yolo_model.trainable_variables):
    print(i,layer.name)


# In[16]:


yolo_model.view_model()


# In[17]:


grads_conf_1 = []
for layer in yolo_model.trainable_variables[:MTL_NO_OF_SHARED_LAYERS]:
    grads_conf_1+=[np.zeros([MTL_GRADS_QLEN]+layer.shape,dtype=np.float32)]
    
grads_conf_2 = []
for layer in yolo_model.trainable_variables[:MTL_NO_OF_SHARED_LAYERS]:
    grads_conf_2+=[np.zeros([MTL_GRADS_QLEN]+layer.shape,dtype=np.float32)]

prior_loss_wts_matrix = [tf.Variable([1.00,1.00]),
                         tf.Variable([1.00,1.00]),
                         tf.Variable([1.00,1.00]),
                         tf.Variable([1.00,1.00])]#*MTL_NO_OF_BLOCKS

loss_rate_matrix = np.array([1.0,1.0])


# In[18]:


n1 = np.prod(yolo_model.trainable_variables[0].numpy().shape)
n2 = np.prod(yolo_model.trainable_variables[3].numpy().shape)
n3 = np.prod(yolo_model.trainable_variables[6].numpy().shape)
n4 = np.prod(yolo_model.trainable_variables[9].numpy().shape)
n = n1+n2+n3+n4
n1/n,n2/n,n3/n,n4/n


# In[19]:


if DEBUG_MODE == True:
    no_train_batch = 20
    no_val_batch   = 3
    TRAIN_EPOCHS   = 20


# In[20]:


if data_gen == True:
    print(len(trainset),len(testset))
    
    os.system(f"rm -r {TRAIN_DATA_SAVE_PATH}*")
    for ind in trange(no_train_batch):
        train_img, train_label  = next(trainset)
        path=TRAIN_DATA_SAVE_PATH+'batch_{}.npy'.format(str(ind))
        for _i in range(NO_OF_GRID):
            train_label[_i][0] = zlib.compress(train_label[_i][0])
        np.save(path,[np.uint8(train_img*255.0),train_label]) 
        
    os.system(f"rm -r {TEST_DATA_SAVE_PATH}*")
    for ind in trange(no_val_batch):
        test_img, test_label = next(testset)
        path=TEST_DATA_SAVE_PATH+'batch_{}.npy'.format(str(ind))
        for _i in range(NO_OF_GRID):
            test_label[_i][0] = zlib.compress(test_label[_i][0])
        np.save(path,[np.uint8(test_img*255.0),test_label])
        
    del trainset, testset


# In[21]:


yolo_model.summary()


# In[22]:


l1 = len(yolo_model.trainable_variables)
if TRAIN_FREEZE_EPOCH>0:
    for ind,var in enumerate(yolo_model.layers):   
        if'block' in var.name:
            yolo_model.layers[ind].trainable = False
                          
l2 = len(yolo_model.trainable_variables)
NO_FROZEN_LAYERS = l1-l2


# In[23]:


yolo_model.summary()


# In[24]:


def seg_to_dst(batch_seg):
    batch_dst= np.uint8(255*(batch_seg!=80))
    for i in range(len(batch_dst)):
        batch_dst[i] = cv2.distanceTransform(batch_dst[i],cv2.DIST_L2,3)
    batch_dst = np.float32(batch_dst/ np.sqrt(128**2 + 128**2))
    return batch_dst

#@tf.function
def get_mini_batch(image,target,ind,mbatch_size):
    ind1,ind2 = ind*mbatch_size,(ind+1)*mbatch_size
    mbatch_target  = [[elem[ind1:ind2] for elem in target[0]],                      [elem[ind1:ind2] for elem in target[1]],                      [elem[ind1:ind2] for elem in target[2]]]
    mbatch_target += [elem[ind1:ind2] for elem in target[3:]]
    return image[ind1:ind2], mbatch_target

if not os.path.exists(TRAIN_CHECKPOINTS_FOLDER): os.makedirs(TRAIN_CHECKPOINTS_FOLDER)
os.system(f'mkdir -p {TRAIN_CHECKPOINTS_FOLDER}/model')
os.system(f'mkdir -p {TRAIN_CHECKPOINTS_FOLDER}/mvars')

def get_loss_wts():
    return np.array([list(mat.numpy()) for mat in prior_loss_wts_matrix]).flatten()

def update_prior():
    _sum = loss_rate_matrix.sum() 
    x,y = loss_rate_matrix/ _sum
    
loss_rate_df = pd.DataFrame(loss_rate_matrix).T
init_loss = 0
epoch = -1
learning_rates      = np.linspace(TRAIN_LR/10,TRAIN_LR,TRAIN_WARM_UP_EPOCHS*no_train_batch)
#mtl_loss_wts_values = np.linspace(MTL_LOSS_WTS[2]/10,MTL_LOSS_WTS[2],TRAIN_WARM_UP_EPOCHS*no_train_batch)
_frozen= TRAIN_FREEZE_EPOCH>0

def np_one_hot():
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size),a] = 1

OUTPUT_SIZES = TRAIN_INPUT_SIZE // np.array(YOLO_STRIDES)
def decompress(x,shape=(16,14,14,3,85)):
    x = zlib.decompress(x)
    x = np.frombuffer(x,dtype=np.float32)
    x = x.reshape(shape)
    return x

while 1:
    epoch +=1
    if os.path.exists('params.json'):
        with open('params.json','r') as file:
            param_dict = json.load(file)
        globals().update(param_dict)
    
    if _frozen==True and epoch>=TRAIN_FREEZE_EPOCH:
        for ind,layer in enumerate(yolo_model.layers):   
            if 'block' in layer.name:
                yolo_model.layers[ind].trainable = True
            print(ind,layer.name,layer.trainable)
            yolo_model.summary()
            _frozen=False
            
    if epoch >= TRAIN_EPOCHS:
        break
 
    all_logs = [] ; sample_losses =[]
    train_loss_dict=loss_dict_obj(); val_loss_dict = loss_dict_obj(); loss_dict={'Epoch' : epoch}
    
    batch_seg=[];batch_dst=[]
    for batch_ind in tqdm(range(no_train_batch)):
        
        tstep = (epoch)*no_train_batch + batch_ind
        if tstep < len(learning_rates):
            optimizer.learning_rate = learning_rates[tstep]
            #MTL_LOSS_WTS[2] = mtl_loss_wts_values[tstep]
            
        if TRAIN_INPUT_SIZE <TRAIN_SAVE_THR_SIZE :
            path=TRAIN_DATA_SAVE_PATH+'batch_{}.npy'.format(str(batch_ind))
            batch_data = np.load(path,allow_pickle=True)
            batch_data[0] = np.float32(batch_data[0]/255.0)
            for _i in range(NO_OF_GRID):
                shape = (TRAIN_BATCH_SIZE,OUTPUT_SIZES[_i],OUTPUT_SIZES[_i],3,NUM_CLASS+5)
                batch_data[1][_i][0] = decompress(batch_data[1][_i][0],shape)
               
        else:
            batch_data=next(trainset)
        
        image_data, target = batch_data[0],list(batch_data[1])+[batch_dst,batch_seg]
        train_loss,ret_det,ret_seg , norms_matrix= train_batch(image_data, target,frozen=_frozen)
        del image_data, target, batch_data
        
        train_loss_dict.sum_update(train_loss)
        all_logs+=[list(train_loss.values())]
        if type(init_loss) == int and init_loss== 0:
            init_loss = np.array([train_loss['iou_loss']+train_loss['conf_loss'],train_loss['prob_loss']])
        else:
            curr_loss = np.array([train_loss['iou_loss']+train_loss['conf_loss'],train_loss['prob_loss']])
            
            loss_rate_matrix = curr_loss / init_loss
            loss_rate_df.loc[int(epoch*no_train_batch + batch_ind)] = loss_rate_matrix
        
        
    train_loss_dict.divide(no_train_batch)
    iou_val, conf_val, prob_val, total_val = 0, 0, 0, 0
    batch_seg=[];batch_dst=[]
    
    for batch_ind in tqdm(range(no_val_batch)):
        if TRAIN_INPUT_SIZE < TRAIN_SAVE_THR_SIZE:
            path=TEST_DATA_SAVE_PATH+'batch_{}.npy'.format(str(batch_ind))
            batch_data=np.load(path,allow_pickle=True)
            batch_data[0] = np.float32(batch_data[0]/255.0)
            for _i in range(NO_OF_GRID):
                shape = (TEST_BATCH_SIZE,OUTPUT_SIZES[_i],OUTPUT_SIZES[_i],3,NUM_CLASS+5)
                batch_data[1][_i][0] = decompress(batch_data[1][_i][0],shape)
        else:
            batch_data = next(testset)
        
        image_data, target = batch_data[0],list(batch_data[1])+[batch_dst,batch_seg]
        val_loss = validate_step(image_data, target)
        del image_data, target, batch_data
        val_loss_dict.sum_update(val_loss)
        
    val_loss_dict.divide(no_val_batch)
    loss_dict.update(train_loss_dict)
    loss_dict.update(val_loss_dict)
    print(loss_dict)
    
     
    if epoch % TRAIN_SAVE_WEIGHTS_EVERY == 0:
        save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER,'model', 'epoch_{}_val_det_loss_{}'.format(epoch,round(loss_dict['val_det_loss'],4)))
        yolo_model.save_weights(save_directory+'/weights')
        save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER,'mvars', f'epoch_{epoch}.pkl')
        with open(save_directory,'wb') as file:
            pickle.dump([grads_conf_1,grads_conf_2],file)

        save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, f'loss_rate_matrix.csv')
        loss_rate_df.to_csv(save_directory)

    if TRAIN_SAVE_BEST_ONLY and best_val_loss>loss_dict['val_det_loss']:
        save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, 'model')
        yolo_model.save_weights(save_directory+'/weights')
        best_val_loss = loss_dict['val_det_loss']

    save_loss_logs(loss_dict,epoch)
    save_std_logs(train_loss,all_logs,epoch)
    #save_sample_losses(sample_losses,epoch)


# In[25]:


OUTPUT_SIZES


# In[26]:


(64*14*14*3*85)


# In[27]:


#yolo_model.trainable_variables


# alpha = 1.0
# rows = 224
# url          = 'https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/'
# model_name   = ('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' +str(float(alpha)) + '_' + str(rows) + '.h5')
# weight_path  = url + model_name
# #weights_path = data_utils.get_file(model_name, weight_path, cache_subdir='models')
