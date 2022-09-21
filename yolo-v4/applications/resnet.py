"""ResNet models for Keras.

Reference:
  - [Deep Residual Learning for Image Recognition](
      https://arxiv.org/abs/1512.03385) (CVPR 2015)
"""

import tensorflow as tf

from tensorflow.keras import backend
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.utils import get_file
from tensorflow.keras.utils import get_source_inputs
from tensorflow.python.util.tf_export import keras_export


BASE_WEIGHTS_PATH = (
    'https://storage.googleapis.com/tensorflow/keras-applications/resnet/')
WEIGHTS_HASHES = {
    'resnet50': ('2cb95161c43110f7111970584f804107',
                 '4d473c1dd8becc155b73f8504c6f6626'),
    'resnet101': ('f1aeb4b969a6efcfb50fad2f0c20cfc5',
                  '88cf7a10940856eca736dc7b7e228a21'),
    'resnet152': ('100835be76be38e30d865e96f2aaae62',
                  'ee4c566cf9a93f14d82f913c2dc6dd0c'),
    'resnet50v2': ('3ef43a0b657b3be2300d5770ece849e0',
                   'fac2f116257151a9d068a22e544a4917'),
    'resnet101v2': ('6343647c601c52e1368623803854d971',
                    'c0ed64b8031c3730f411d2eb4eea35b5'),
    'resnet152v2': ('a49b44d1979771252814e80f8ec446f9',
                    'ed17cf2e0169df9d443503ef94b23b33'),
    'resnext50': ('67a5b30d522ed92f75a1f16eef299d1a',
                  '62527c363bdd9ec598bed41947b379fc'),
    'resnext101':
        ('34fb605428fcc7aa4d62f44404c11509', '0f678c91647380debd923963594981b3')
}

layers = tf.keras.layers

def ResNet(stack_fn,
           use_bias,
           model_name='resnet',
           include_top=True,
           weights='imagenet',
           input_shape=None,
           pooling=None,
           classes=1000,
           classifier_activation='softmax',
           **kwargs):
    
    img_input = layers.Input(shape=input_shape)
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)
  
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
    x = layers.Activation('relu', name='conv1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = stack_fn(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation=classifier_activation,name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)
    
    inputs = img_input

    # Create model.
    model = tf.keras.models.Model(inputs, x, name=model_name)

    # Load weights.
    if (weights == 'imagenet') and (model_name in WEIGHTS_HASHES):
        if include_top:
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels.h5'
            file_hash = WEIGHTS_HASHES[model_name][0]
        else:
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_notop.h5'
            file_hash = WEIGHTS_HASHES[model_name][1]
        weights_path = get_file(
            file_name,
            BASE_WEIGHTS_PATH + file_name,
            cache_subdir='models',
            file_hash=file_hash)
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None, mini=False):
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x 
        
    x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)
    
    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)    

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x

def block2(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None, mini=False):
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x 

    x = layers.Conv2D(filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)
    
    x = layers.Conv2D(4 * filters, kernel_size, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)
        

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack1(x, filters, blocks, stride=2, name=None,mini=False):
    x = block1(x, filters, stride=stride, name=name + '_block1',mini=mini)
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i),mini=mini)
    return x

def ResNet18(include_top=True,
             weights='imagenet',
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
    def stack_fn(x):
        x = stack1(x, 64,  2, stride=1, name='conv2',mini=True)
        x = stack1(x, 128, 2, stride=2, name='conv3',mini=True)
        x = stack1(x, 256, 2, stride=2, name='conv4',mini=True)
        x = stack1(x, 512, 2, stride=2, name='conv5',mini=True)
        return x

    return ResNet(stack_fn, True, 'resnet50', include_top, weights, input_shape, pooling, classes, **kwargs)

def ResNet34(include_top=True,
             weights='imagenet',
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
    def stack_fn(x):
        x = stack1(x, 64,  3, stride=1, name='conv2',mini=True)
        x = stack1(x, 128, 4, stride=2, name='conv3',mini=True)
        x = stack1(x, 256, 6, stride=2, name='conv4',mini=True)
        x = stack1(x, 512, 2, stride=2, name='conv5',mini=True)
        return x

    return ResNet(stack_fn, True, 'resnet50', include_top, weights, input_shape, pooling, classes, **kwargs)


def ResNet50(include_top=True,
             weights='imagenet',
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
    def stack_fn(x):
        x = stack1(x, 64,  3, stride=1, name='conv2')
        x = stack1(x, 128, 4, stride=2, name='conv3')
        x = stack1(x, 256, 6, stride=2, name='conv4')
        x = stack1(x, 512, 3, stride=2, name='conv5')
        return x

    return ResNet(stack_fn, True, 'resnet50', include_top, weights, input_shape, pooling, classes, **kwargs)


def ResNet101(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000,
              **kwargs):

    def stack_fn(x):
        x = stack1(x, 64,  3,  stride=1, name='conv2')
        x = stack1(x, 128, 4,  stride=2,name='conv3')
        x = stack1(x, 256, 23, stride=2,  name='conv4')
        x = stack1(x, 512, 3,  stride=2, name='conv5')
        return x

    return ResNet(stack_fn, False, True, 'resnet101', include_top, weights,input_tensor, input_shape, pooling, classes, **kwargs)

def ResNet152(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000,
              **kwargs): 
    def stack_fn(x):
        x = stack1(x, 64,  3,  stride1=1, name='conv2')
        x = stack1(x, 128, 8,  stride1=2, name='conv3')
        x = stack1(x, 256, 36, stride1=2, name='conv4')
        x = stack1(x, 512, 3,  stride1=2, name='conv5')
        return x
    
    return ResNet(stack_fn, False, True, 'resnet152', include_top, weights,input_tensor, input_shape, pooling, classes, **kwargs)


def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(x, data_format=data_format, mode='caffe')


def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)


'''
def stack2(x, filters, blocks, stride1=2, name=None):

    x = block2(x, filters, conv_shortcut=True, name=name + '_block1')
    for i in range(2, blocks):
        x = block2(x, filters, name=name + '_block' + str(i))
    x = block2(x, filters, stride=stride1, name=name + '_block' + str(blocks))
    return x


def block2(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    preact = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_preact_bn')(x)
    preact = layers.Activation('relu', name=name + '_preact_relu')(preact)

    if conv_shortcut:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride, name=name + '_0_conv')(preact)
    else:
        shortcut = layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

    x = layers.Conv2D(
      filters, 1, strides=1, use_bias=False, name=name + '_1_conv')(preact)
    x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = layers.Conv2D(filters,kernel_size,strides=stride,use_bias=False,name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)
    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.Add(name=name + '_out')([shortcut, x])
    return x



def block3(x,
           filters,
           kernel_size=3,
           stride=1,
           groups=32,
           conv_shortcut=True,
           name=None):
  
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut:
        shortcut = layers.Conv2D(
        (64 // groups) * filters,
        1,
        strides=stride,
        use_bias=False,
        name=name + '_0_conv')(x)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, use_bias=False, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    c = filters // groups
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        depth_multiplier=c,
        use_bias=False,
        name=name + '_2_conv')(x)
    x_shape = backend.shape(x)[:-1]
    x = backend.reshape(x, backend.concatenate([x_shape, (groups, c, c)]))
    x = layers.Lambda(
      lambda x: sum(x[:, :, :, :, i] for i in range(c)),
      name=name + '_2_reduce')(x)
    x = backend.reshape(x, backend.concatenate([x_shape, (filters,)]))
    x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(
      (64 // groups) * filters, 1, use_bias=False, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack3(x, filters, blocks, stride1=2, groups=32, name=None):
    x = block3(x, filters, stride=stride1, groups=groups, name=name + '_block1')
    for i in range(2, blocks + 1):
    x = block3(
        x,
        filters,
        groups=groups,
        conv_shortcut=False,
        name=name + '_block' + str(i))
    return x

'''