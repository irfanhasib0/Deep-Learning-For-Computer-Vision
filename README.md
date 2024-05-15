<a id=table_of_content></a>
# Deep-Learning-For-Computer-Vision
Most of my [works](https://irfanhasib0.github.io) before 2022 were implementation from scratch. Now a days every day new papers are being publishesd. It's impossible to implement each them from scratch. For catching up with SOTA algorithms sometimes I work on top of the publicly available source code of respective researches.

- <a href='#abn_motion'> Abnormal  Motion Detection (From Scratch)</a>
- <a href='#pose_track'> Pose Detection and Tracking</a>
- <a href='#sem_seg'>Semantc segmentation with DeepLabV3</a>
- <a href='#yolov8'>Yolo-v8 for object detection </a>
- <a href='#transformer'> Transformer for translation (From Scratch)</a>
- <a href='#diffusion'> Diffusion model for car image generation.</a>

<a id='abn_motion'></a>
## Abnormal Motion Detection (Implementation From scratch)
<a href='#table_of_content'> back </a>

  - Code : [motion-anomaly](https://github.com/irfanhasib0/Deep-Learning-For-Computer-Vision/blob/main/motion-anomaly/)
  - src/stream_io.py : Takes care of simultanous videoo and audio capture from webcam / ipcam / video file using ffmpeg.
  - src/feat_tracker.py : Detects key feature points and tracks them.
              - Feature Points : good feature to track / human pose key points
              - Tracker : optical flow based tracker / pose flow tracker
  - src/gui.py : Visualize audio intensity and feature point velocity.

### Results
<img src=https://github.com/irfanhasib0/Deep-Learning-For-Computer-Vision/blob/main/motion-anomaly/results/lock_1.gif align='left' width='100%'>
```
```

<a id='pose_track'></a>
## Pose Tracking (Working on top of existing public source code)
<a href='#table_of_content'> back </a>
  - Code : [pose-tracking](https://github.com/irfanhasib0/Deep-Learning-For-Computer-Vision/blob/main/pose_tracking/)
  - posedet_mnet  : [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) with mobilenet
  - posedet_alpha : [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)
  - posedet_alpha/tracker/Pose Flow : Pose Flow tracker.

### Results
<img src = https://github.com/irfanhasib0/Deep-Learning-For-Computer-Vision/blob/main/pose-tracking/results/result_3_50.gif align='left' width='100%'>
```
```

<a id='sem_seg'></a>
## DeeplabV3 (Working on top of existing public source code)
<a href='#table_of_content'> back </a>

  - Code : [deeplabv3](https://github.com/irfanhasib0/Deep-Learning-For-Computer-Vision/blob/main/deeplabv3/)
  Inspired from ![deeplabv3plus](https://github.com/VainF/DeepLabV3Plus-Pytorch)
  - I implemented UNet from scratch here ![UNet](https://github.com/irfanhasib0/CNN-Projects/blob/master/CNN_Basic/U-Net_cityscapes.ipynb)
  In this repository I am working on DeeplabV3. Here are some resuts of training on VOC Dataset. I have borrowed some code from the internet since now a days lot's of technology are coming every day. It's not practical to implement everything from scratch and stay up to date at the same time.
<div class='row' width='100%'>
<img src=https://github.com/irfanhasib0/Deep-Learning-For-Computer-Vision/blob/main/deeplabv3/results/result_3.png align='left' width="47%">
<img src=https://github.com/irfanhasib0/Deep-Learning-For-Computer-Vision/blob/main/deeplabv3/results/result_4.png align='left' width="48%">
</div>
```
```

<a id='#yolov8'></a>
## YOLO-V8 (Working on top of public source code)
<a href='#table_of_content'> back </a>
  - yolo-v8 is a modified version of publicly available source code of [Ultralytics](https://github.com/ultralytics/ultralytics)
  - Code : [yolo-v8](https://github.com/irfanhasib0/Deep-Learning-For-Computer-Vision/blob/main/yolo-v8/)
  - yolo-v4 is my personal implementation of yolo v4 from scratch. It can achieve aroung mAP 25 with mobilenet with alpha = 1 and image size 224x224. I have taken help from other resouces available in the internet for this work.
  - Code : [yolo-v4](https://github.com/irfanhasib0/Deep-Learning-For-Computer-Vision/blob/main/yolo-v4/) ; 
  [notebook](https://github.com/irfanhasib0/CNN-Projects/blob/master/CNN_Basic/Minimal_yolo_coco-v-2.0-exp-COCO.ipynb)
### Results
  <img src='https://github.com/irfanhasib0/Deep-Learning-For-Computer-Vision/blob/main/yolo-v8/results/val_batch0_pred.jpg' align='left' width='100%'>
```
```

<a id='#transformer'></a>
## Transformer
<a href='#table_of_content'> back </a>
<p> A simple transformer has been trained on <a href='https://arxiv.org/abs/1605.00459'>Multi30k</a> dataset for German to English transation. </p>

Code : [transformer-models](https://github.com/irfanhasib0/Deep-Learning-For-Computer-Vision/blob/main/transformer-models/)
### Usage :
#### Main note book 
  - transforer/transformer.ipynb 
#### Implementation from scratch 
  -transformer/models/transformer_v1.py 
#### Implementation using pytorch nn.MultiHeadAttention Module
  -transformer/models/transformer_v2.py

Note : vision-transformer/ is a on progress work. The code is not organized at all.

- Results:

```
Epoch: 1, Train loss: 5.342, Val loss: 4.104, Epoch time = 51.305s
Epoch: 2, Train loss: 3.759, Val loss: 3.306, Epoch time = 55.960s
Epoch: 3, Train loss: 3.156, Val loss: 2.888, Epoch time = 65.804s
Epoch: 4, Train loss: 2.767, Val loss: 2.629, Epoch time = 71.922s
Epoch: 5, Train loss: 2.478, Val loss: 2.442, Epoch time = 74.421s
Epoch: 6, Train loss: 2.249, Val loss: 2.307, Epoch time = 72.754s
Epoch: 7, Train loss: 2.056, Val loss: 2.217, Epoch time = 78.290s
Epoch: 8, Train loss: 1.895, Val loss: 2.108, Epoch time = 76.270s
Epoch: 9, Train loss: 1.754, Val loss: 2.053, Epoch time = 79.638s
Epoch: 10, Train loss: 1.632, Val loss: 1.996, Epoch time = 83.422s
Epoch: 11, Train loss: 1.523, Val loss: 1.965, Epoch time = 87.027s
Epoch: 12, Train loss: 1.418, Val loss: 1.939, Epoch time = 86.907s
Epoch: 13, Train loss: 1.328, Val loss: 1.928, Epoch time = 90.075s
Epoch: 14, Train loss: 1.250, Val loss: 1.940, Epoch time = 96.738s
Epoch: 15, Train loss: 1.172, Val loss: 1.936, Epoch time = 96.887s
Epoch: 16, Train loss: 1.101, Val loss: 1.915, Epoch time = 97.977s
Epoch: 17, Train loss: 1.035, Val loss: 1.895, Epoch time = 97.573s
Epoch: 18, Train loss: 0.976, Val loss: 1.911, Epoch time = 97.933s
```

### Sample Result of trained model
- Input (German)   : Eine Gruppe von Menschen steht vor einem Iglu . 
- Output (English) : A group of people standing in front of an igloo .
```
```

<a id='#diffusion'></a>
## Diffusion
<a href='#table_of_content'> back </a>
<p>A simple diffusion model has been trained on car images.</p>

 - Code : [diffusion-models](https://github.com/irfanhasib0/Deep-Learning-For-Computer-Vision/blob/main/diffusion-models/)

### Results
<img src='https://github.com/irfanhasib0/Deep-Learning-For-Computer-Vision/blob/main/diffusion-models/results/diff_car_1.png' align='left' width='100%'>
```
```


