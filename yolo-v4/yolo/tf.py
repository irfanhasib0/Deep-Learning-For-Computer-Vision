from .config import *
import os
os.environ['PYTHONHASHSEED']=str(0)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import tensorflow as tf
#from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU, ZeroPadding2D, BatchNormalization, MaxPool2D,\
                                    Cropping2D, UpSampling2D, Add, Softmax, Conv2DTranspose, concatenate,\
                                    GlobalAveragePooling2D, Reshape, Dense, Permute, multiply, ReLU
from tensorflow.keras.activations import sigmoid as Sigmoid
from tensorflow.keras.regularizers import l2

tf.random.set_seed(SEED)
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    