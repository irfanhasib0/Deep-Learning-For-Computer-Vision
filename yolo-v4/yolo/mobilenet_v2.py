# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=invalid-name
"""MobileNet v2 models for Keras.

MobileNetV2 is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and
different width factors. This allows different width models to reduce
the number of multiply-adds and thereby
reduce inference cost on mobile devices.

MobileNetV2 is very similar to the original MobileNet,
except that it uses inverted residual blocks with
bottlenecking features. It has a drastically lower
parameter count than the original MobileNet.
MobileNets support any input size greater
than 32 x 32, with larger image sizes
offering better performance.

The number of parameters and number of multiply-adds
can be modified by using the `alpha` parameter,
which increases/decreases the number of filters in each layer.
By altering the image size and `alpha` parameter,
all 22 models from the paper can be built, with ImageNet weights provided.

The paper demonstrates the performance of MobileNets using `alpha` values of
1.0 (also called 100 % MobileNet), 0.35, 0.5, 0.75, 1.0, 1.3, and 1.4
For each of these `alpha` values, weights for 5 different input image sizes
are provided (224, 192, 160, 128, and 96).

The following table describes the performance of
MobileNet on various input sizes:
------------------------------------------------------------------------
MACs stands for Multiply Adds
 Classification Checkpoint|MACs (M)|Parameters (M)|Top 1 Accuracy|Top 5 Accuracy
--------------------------|------------|---------------|---------|----|---------
| [mobilenet_v2_1.4_224]  | 582 | 6.06 |          75.0 | 92.5 |
| [mobilenet_v2_1.3_224]  | 509 | 5.34 |          74.4 | 92.1 |
| [mobilenet_v2_1.0_224]  | 300 | 3.47 |          71.8 | 91.0 |
| [mobilenet_v2_1.0_192]  | 221 | 3.47 |          70.7 | 90.1 |
| [mobilenet_v2_1.0_160]  | 154 | 3.47 |          68.8 | 89.0 |
| [mobilenet_v2_1.0_128]  | 99  | 3.47 |          65.3 | 86.9 |
| [mobilenet_v2_1.0_96]   | 56  | 3.47 |          60.3 | 83.2 |
| [mobilenet_v2_0.75_224] | 209 | 2.61 |          69.8 | 89.6 |
| [mobilenet_v2_0.75_192] | 153 | 2.61 |          68.7 | 88.9 |
| [mobilenet_v2_0.75_160] | 107 | 2.61 |          66.4 | 87.3 |
| [mobilenet_v2_0.75_128] | 69  | 2.61 |          63.2 | 85.3 |
| [mobilenet_v2_0.75_96]  | 39  | 2.61 |          58.8 | 81.6 |
| [mobilenet_v2_0.5_224]  | 97  | 1.95 |          65.4 | 86.4 |
| [mobilenet_v2_0.5_192]  | 71  | 1.95 |          63.9 | 85.4 |
| [mobilenet_v2_0.5_160]  | 50  | 1.95 |          61.0 | 83.2 |
| [mobilenet_v2_0.5_128]  | 32  | 1.95 |          57.7 | 80.8 |
| [mobilenet_v2_0.5_96]   | 18  | 1.95 |          51.2 | 75.8 |
| [mobilenet_v2_0.35_224] | 59  | 1.66 |          60.3 | 82.9 |
| [mobilenet_v2_0.35_192] | 43  | 1.66 |          58.2 | 81.2 |
| [mobilenet_v2_0.35_160] | 30  | 1.66 |          55.7 | 79.1 |
| [mobilenet_v2_0.35_128] | 20  | 1.66 |          50.8 | 75.0 |
| [mobilenet_v2_0.35_96]  | 11  | 1.66 |          45.5 | 70.4 |

  Reference:
  - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](
      https://arxiv.org/abs/1801.04381) (CVPR 2018)
"""
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
'''
from keras import backend
from keras.applications import imagenet_utils
from keras.engine import training
from keras.layers import VersionAwareLayers

from keras.utils import layer_utils
'''

BASE_WEIGHT_PATH = ('https://storage.googleapis.com/tensorflow/'
                    'keras-applications/mobilenet_v2/')
layers = tf.keras.layers
class MobileNetV2(tf.keras.Model):
    def __init__(self,input_shape=None,
                    alpha        =1.0,
                    input_tensor=None,
                    pooling     =None,
                    NUM_CLASS   =80,
                    N           =1,
                    NH          =1,
                    load_weights=True,
                    OBJ_SCALES  = ['small','medium','large']):
        super(MobileNetV2,self).__init__()
        self.N = N
        self.NH = NH
        self.alpha = alpha
        self.obj_scales = OBJ_SCALES
        self._inverted_res_block = _inverted_res_block
        filters = _make_divisible(32 * alpha, 8)
        self.conv2d_1 = layers.Conv2D(filters,input_shape=input_shape,kernel_size=3,strides=(2, 2),padding='same',use_bias=False,name='block0_conv1')
        self.bn_1     = layers.BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name='block0_bn_conv1')
        self.relu_1   = layers.ReLU(6., name='block0_conv1_relu')
        
        channels = filters; filters = 16/N
        self.inv_res_block_0 = _inverted_res_block(filters=filters, channels=channels, alpha=alpha, stride=1, expansion=1, block_id=0)
        channels = filters; filters = 24/N
        self.inv_res_block_1 = _inverted_res_block(filters=filters, channels=channels, alpha=alpha, stride=2, expansion=6, block_id=1)
        channels = filters; filters = 24/N
        self.inv_res_block_2 = _inverted_res_block(filters=filters, channels=channels, alpha=alpha, stride=1, expansion=6, block_id=2)
        
        channels = filters; filters = 32/N
        self.inv_res_block_3 = _inverted_res_block(filters=filters, channels=channels, alpha=alpha, stride=2, expansion=6, block_id=3)
        channels = filters; filters = 32/N
        self.inv_res_block_4 = _inverted_res_block(filters=filters, channels=channels, alpha=alpha, stride=1, expansion=6, block_id=4)
        channels = filters; filters = 32/N
        self.inv_res_block_5 = _inverted_res_block(filters=filters, channels=channels, alpha=alpha, stride=1, expansion=6, block_id=5)
        
        channels = filters; filters = 64/N
        self.inv_res_block_6 = _inverted_res_block(filters=filters, channels=channels, alpha=alpha, stride=2, expansion=6, block_id=6)
        channels = filters; filters = 64/N
        self.inv_res_block_7 = _inverted_res_block(filters=filters, channels=channels, alpha=alpha, stride=1, expansion=6, block_id=7)
        channels = filters; filters = 64/N
        self.inv_res_block_8 = _inverted_res_block(filters=filters, channels=channels, alpha=alpha, stride=1, expansion=6, block_id=8)
        channels = filters; filters = 64/N
        self.inv_res_block_9 = _inverted_res_block(filters=filters, channels=channels, alpha=alpha, stride=1, expansion=6, block_id=9)
        
        channels = filters; filters = 96/N
        self.inv_res_block_10 = _inverted_res_block(filters=filters, channels=channels, alpha=alpha, stride=1, expansion=6, block_id=10)
        channels = filters; filters = 96/N
        self.inv_res_block_11 = _inverted_res_block(filters=filters, channels=channels, alpha=alpha, stride=1, expansion=6, block_id=11)
        channels = filters; filters = 96/N
        self.inv_res_block_12 = _inverted_res_block(filters=filters, channels=channels, alpha=alpha, stride=1, expansion=6, block_id=12)
        
        channels = filters; filters = 160/N
        self.inv_res_block_13 = _inverted_res_block(filters=filters, channels=channels, alpha=alpha, stride=2, expansion=6, block_id=13)
        channels = filters; filters = 160/N
        self.inv_res_block_14 = _inverted_res_block(filters=filters, channels=channels, alpha=alpha, stride=1, expansion=6, block_id=14)
        channels = filters; filters = 160/N
        self.inv_res_block_15 = _inverted_res_block(filters=filters, channels=channels, alpha=alpha, stride=1, expansion=6, block_id=15)
        channels = filters; filters = 320/N
        self.inv_res_block_16 = _inverted_res_block(filters=filters, channels=channels, alpha=alpha, stride=1, expansion=6, block_id=16)
         
        if alpha > 1.0:
            last_block_filters = _make_divisible(1280 * alpha, 8)
        else:
            last_block_filters = 1280
        self.conv2d_2 = layers.Conv2D(last_block_filters, kernel_size=1, use_bias=False, name='block_ext_vonv_1')
        self.bn_2     = layers.BatchNormalization(axis=-1, epsilon=1e-3, momentum=0.999, name='block_ext_conv_1_bn')
        self.relu_2   = layers.ReLU(6., name='block_ext_out_relu')
    
        self.conv_lobj_1 = layers.Conv2D(filters=256//self.NH, kernel_size = 1, strides=1,
                          padding='same', use_bias=False, kernel_regularizer=l2(0.0005),
                          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                          bias_initializer=tf.constant_initializer(0.),name='conv_'+str(1))
        self.bn_lobj_1   = layers.BatchNormalization()
        self.relu_lobj_1 = layers.LeakyReLU(alpha=0.1)
        
        self.conv_lobj_2 = layers.Conv2D(filters=512//self.NH, kernel_size = 3, strides=1,
                          padding='same', use_bias=False, kernel_regularizer=l2(0.0005),
                          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                          bias_initializer=tf.constant_initializer(0.),name='conv_'+str(2))
        self.bn_lobj_2   = layers.BatchNormalization()
        self.relu_lobj_2 = layers.LeakyReLU(alpha=0.1)
        #conv_lobj_branch = self.convolutional(conv, (3, 3, 8*self.N, 16*self.N))
        
        self.conv_lobj_3 = layers.Conv2D(filters=3*(NUM_CLASS + 5), kernel_size = 1, strides=1,
                          padding='same', use_bias=False, kernel_regularizer=l2(0.0005),
                          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                          bias_initializer=tf.constant_initializer(0.),name='conv_'+str(3))
        #conv_lbbox = self.convolutional(conv_lobj_branch, (1, 1, 16*self.N, 3*(NUM_CLASS + 5)), activate=False, bn=False)
        if 'medium' in self.obj_scales:
            self.conv_mobj_1 = layers.Conv2D(filters=128//self.NH, kernel_size = 1, strides=1,
                              padding='same', use_bias=False, kernel_regularizer=l2(0.0005),
                              kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                              bias_initializer=tf.constant_initializer(0.),name='conv_'+str(4))
            self.bn_mobj_1   = layers.BatchNormalization()
            self.relu_mobj_1 = layers.LeakyReLU(alpha=0.1)
            #conv = self.convolutional(conv, (1, 1, 8*self.N, 4*self.N))

            self.ups_mobj_1 = layers.Conv2DTranspose(128//self.NH,(3,3),strides =(2,2),padding='same')
            #conv = self.upsample(conv)
            self.concat_mobj_1 = layers.concatenate#([conv, route_2], axis=-1)

            self.conv_mobj_2 = layers.Conv2D(filters=256//self.NH, kernel_size = 3, strides=1,
                              padding='same', use_bias=False, kernel_regularizer=l2(0.0005),
                              kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                              bias_initializer=tf.constant_initializer(0.),name='conv_'+str(5))
            self.bn_mobj_2   = layers.BatchNormalization()
            self.relu_mobj_2 = layers.LeakyReLU(alpha=0.1)
            #conv_mobj_branch = self.convolutional(conv, (3, 3, 8*self.N, 16*self.N))

            self.conv_mobj_3 = layers.Conv2D(filters=3 * (NUM_CLASS + 5), kernel_size = 3, strides=1,
                              padding='same', use_bias=False, kernel_regularizer=l2(0.0005),
                              kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                              bias_initializer=tf.constant_initializer(0.),name='conv_'+str(6))
            #conv_mbbox = self.convolutional(conv_mobj_branch, (1, 1, 16*self.N, 3 * (NUM_CLASS + 5)), activate=False, bn=False)
        if 'small' in self.obj_scales:
            self.conv_sobj_1 = layers.Conv2D(filters=128//self.NH, kernel_size = 1, strides=1,
                              padding='same', use_bias=False, kernel_regularizer=l2(0.0005),
                              kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                              bias_initializer=tf.constant_initializer(0.),name='conv_'+str(7))
            self.bn_sobj_1   = layers.BatchNormalization()
            self.relu_sobj_1 = layers.LeakyReLU(alpha=0.1)
            #conv = self.convolutional(conv, (1, 1, 8*self.N, 4*self.N))

            self.ups_sobj_1 =  layers.Conv2DTranspose(128//self.NH,(3,3),strides =(2,2),padding='same')
            #conv = self.upsample(conv)
            self.concat_sobj_1 = layers.concatenate

            self.conv_sobj_2 = layers.Conv2D(filters=256//self.NH, kernel_size = 3, strides=1,
                              padding='same', use_bias=False, kernel_regularizer=l2(0.0005),
                              kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                              bias_initializer=tf.constant_initializer(0.),name='conv_'+str(8))
            self.bn_sobj_2   = layers.BatchNormalization()
            self.relu_sobj_2 = layers.LeakyReLU(alpha=0.1)
            #conv_sobj_branch = self.convolutional(conv, (3, 3, 8*self.N, 16*self.N))

            self.conv_sobj_3 = layers.Conv2D(filters=3 * (NUM_CLASS + 5), kernel_size = 1, strides=1,
                              padding='same', use_bias=False, kernel_regularizer=l2(0.0005),
                              kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                              bias_initializer=tf.constant_initializer(0.),name='conv_'+str(9))
        #conv_sbbox = self.convolutional(conv_sobj_branch, (1, 1, 16*self.N, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        #return [self.conv_sobj_3, self.conv_mobj_3, self.conv_lobj_3]
    
    def call(self,img_input):
        
        '''
        if input_shape:
            img_input = layers.Input(shape=self.input_shape)
        else:
            img_input = input_tensor
        '''
        N = self.N
        alpha = self.alpha
        
        x = self.conv2d_1(img_input)
        x = self.bn_1(x)
        x = self.relu_1(x)
        
        x = self.inv_res_block_0(x)
        x = self.inv_res_block_1(x)
        x = self.inv_res_block_2(x)
        
        x = self.inv_res_block_3(x)
        x = self.inv_res_block_4(x)
        x = self.inv_res_block_5(x)
        out_1 = x
        x = self.inv_res_block_6(x)
        x = self.inv_res_block_7(x)
        x = self.inv_res_block_8(x)
        x = self.inv_res_block_9(x)
        
        x = self.inv_res_block_10(x)
        x = self.inv_res_block_11(x)
        x = self.inv_res_block_12(x)
        out_2 = x
        x = self.inv_res_block_13(x)
        x = self.inv_res_block_14(x)
        x = self.inv_res_block_15(x)
        
        x = self.inv_res_block_16(x)
        
        conv_out = []
        
        x     = self.conv2d_2(x)
        x     = self.bn_2(x)
        out_3 = self.relu_2(x)
        
        x = self.conv_lobj_1(out_3)
        x = self.bn_lobj_1(x)
        xl = self.relu_lobj_1(x)
        
        x = self.conv_lobj_2(xl)
        x = self.bn_lobj_2(x)
        x = self.relu_lobj_2(x)
        
        conv_lobj = self.conv_lobj_3(x)
        conv_out += [conv_lobj]
        
        if 'medium' in self.obj_scales:
            x = self.conv_mobj_1(xl)
            x = self.bn_mobj_1(x)
            x = self.relu_mobj_1(x)
            x = self.ups_mobj_1(x)
            xm = self.concat_mobj_1([x,out_2],axis=-1)

            x = self.conv_mobj_2(xm)
            x = self.bn_mobj_2(x)
            x = self.relu_mobj_2(x)
            conv_mobj = self.conv_mobj_3(x)
            conv_out += [conv_mobj]
            
        if 'small' in self.obj_scales:
            x = self.conv_sobj_1(xm)
            x = self.bn_sobj_1(x)
            x = self.relu_sobj_1(x)
            x = self.ups_sobj_1(x)
            xs = self.concat_sobj_1([x,out_1],axis=-1)

            x = self.conv_sobj_2(xs)
            x = self.bn_sobj_2(x)
            x = self.relu_sobj_2(x)
            conv_sobj = self.conv_sobj_3(x)
            conv_out+=[conv_sobj]
            
        return conv_out[::-1]
    
    def set_wts(self,wts_path):
        def set_layer_wts(wts,nm):
            for ind,layer in enumerate(self.layers):
                if layer.name in self.updated or nm in self.updated: continue
                if len(layer.get_weights()) and [wt.shape for wt in layer.get_weights()] == [wt.shape for wt in wts]: 
                    self.layers[ind].set_weights(wts)
                    self.updated+=[layer.name,nm]
                    print(f" === setting weights from {nm} to {layer.name}")
                    return
                if 'layers' in self.layers[ind].__dict__:
                    for lind,layer in enumerate(self.layers[ind].layers):
                        if layer.name in self.updated or nm in self.updated: continue
                        if len(layer.get_weights()) and [wt.shape for wt in layer.get_weights()] == [wt.shape for wt in wts]: 
                            self.layers[ind].layers[lind].set_weights(wts)
                            self.updated+=[layer.name,nm]
                            print(f" === setting weights from {nm} to {layer.name}")
                            return
         
        model = load_model(wts_path)
        self.updated = []
        for layer in model.layers:
            wts = layer.get_weights()
            if len(wts)==0:continue
            print(f"getting wts from {layer.name}",end=' ')
            set_layer_wts(wts,layer.name)
            #print('')
            
                   
    def view_model(self):
        for lyr in self.layers:
            try    : print(lyr.name,lyr._trainable_weights[0].shape)
            except : print(lyr.name)
            if 'layers' in lyr.__dict__:
                for lr in lyr.layers:
                    try    : print(lr.name,lr._trainable_weights[0].shape)
                    except : print(lr.name)

        
            
class _inverted_res_block(tf.keras.layers.Layer):
    def __init__(self, expansion=1, stride=1, alpha=1, filters=16, channels=3, block_id=0):
        super(_inverted_res_block,self).__init__(name=f'inv_res_block_{block_id}')
        
        pointwise_filters = int(filters * alpha)
        prefix = 'block_{}_'.format(block_id)
        
        self.layers = []
        self.add_layer = -1
        if block_id:
            self.layers += [layers.Conv2D(expansion * channels,kernel_size=1,padding='same',use_bias=False,activation=None,name=prefix + 'expand')]
            self.layers += [layers.BatchNormalization(axis=-1,epsilon=1e-3,momentum=0.999,name=prefix + 'expand_BN')]
            self.layers += [layers.ReLU(6., name=prefix + 'expand_relu')]
        else:
            prefix = 'block_xx_expanded_conv_'

        self.layers += [layers.DepthwiseConv2D(kernel_size=3,strides=stride,activation=None,use_bias=False,padding='same', name=prefix + 'depthwise')]
        self.layers += [layers.BatchNormalization(axis=-1,epsilon=1e-3,momentum=0.999,name=prefix + 'depthwise_BN')]
        self.layers += [layers.ReLU(6., name=prefix + 'depthwise_relu')]

        self.layers += [layers.Conv2D(pointwise_filters,kernel_size=1,padding='same',use_bias=False,activation=None,name=prefix + 'project')]
        self.layers += [layers.BatchNormalization(axis=-1,epsilon=1e-3,momentum=0.999,name=prefix + 'project_BN')]
        
        if channels == pointwise_filters and stride == 1:
            self.add_layer = layers.Add(name=prefix + 'add')
       
    def call(self,x):
        inputs = x
        for layer in self.layers:
            x = layer(x)
        if type(self.add_layer)!=int:
            x = self.add_layer([inputs,x])
        return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(x, data_format=data_format, mode='tf')


def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)