from .config import *
from .tf import *
import numpy as np
np.random.seed(SEED)
ANCHORS = np.array(ANCHORS, np.float32)/416.0
GRIDS = [TRAIN_INPUT_SIZE//stride for stride in YOLO_STRIDES]

class BatchNormalization(BatchNormalization):
        # "Frozen state" and "inference mode" are two separate concepts.
        # `layer.trainable = False` is to freeze the layer, so the layer will use
        # stored moving `var` and `mean` in the "inference mode", and both `gama`
        # and `beta` will not be updated !
        def call(self, x, training=False):
            if not training:
                training = tf.constant(False)
            training = tf.logical_and(training, self.trainable)
            return super().call(x, training)

class YoloModel():
    def __init__(self,model=YOLO_MODEL,input_size=YOLO_INPUT_SIZE,training=False):
        self._training=training
        self.N = int(TRAIN_MODEL_SCALE) 
        self.model = model
        self.layer_no =0
        print(self.model)
        self.input_size = input_size
        self.mobilenet_v2()
        
    def get_model(self):
        return self.yolo_model
    
    def convolutional(self,input_layer, filters_shape, downsample=False, activate=True, bn=True):
        if downsample:
            input_layer = ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
            padding = 'valid'
            strides = 2
        else:
            strides = 1
            padding = 'same'

        conv = Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides,
                      padding=padding, use_bias=not bn, kernel_regularizer=l2(0.0005),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                      bias_initializer=tf.constant_initializer(0.),name='conv_'+str(self.layer_no))(input_layer)
        if bn:
            conv = BatchNormalization()(conv)
        if activate == True:
            conv = LeakyReLU(alpha=0.1)(conv)
        self.layer_no +=1
        return conv
    
    
    def residual_block(self,input_layer, input_channel, filter_num1, filter_num2):
        short_cut = input_layer
        conv = self.convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1))
        conv = self.convolutional(conv       , filters_shape=(3, 3, filter_num1,   filter_num2))

        residual_output = short_cut + conv
        return residual_output
    
    def squeeze_excite_block(self,init, ratio=8):
        filters = init.shape[-1]
        se_shape = (1, 1, filters)
        se = GlobalAveragePooling2D()(init)
        se = Reshape(se_shape)(se)
        se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
        x = multiply([init, se])
        
        return x
    
    def spatial_attention_block(self,input_tensor, ratio=8):
        avg_pool = tf.reduce_mean(input_tensor,axis=-1)[:,:,:,tf.newaxis]
        sa_out = Conv2D(1,(7,7), padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(avg_pool)
        x = multiply([input_tensor, sa_out])
        return x

    def upsample(self,input_layer):
        return Conv2DTranspose(input_layer.shape[-1],(3,3),strides =(2,2),padding='same')(input_layer)
        #return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')

    
    def darknet53(self,input_data):
        input_data = self.convolutional(input_data, (3, 3,  3,  32))
        input_data = self.convolutional(input_data, (3, 3, 32,  64), downsample=True)

        for i in range(1):
            input_data = self.residual_block(input_data,  64,  32, 64)

        input_data = self.convolutional(input_data, (3, 3,  64, 128), downsample=True)

        for i in range(2):
            input_data = self.residual_block(input_data, 128,  64, 128)

        input_data = self.convolutional(input_data, (3, 3, 128, 256), downsample=True)

        for i in range(8):
            input_data = self.residual_block(input_data, 256, 128, 256)

        route_1 = input_data
        input_data = self.convolutional(input_data, (3, 3, 256, 512), downsample=True)

        for i in range(8):
            input_data = self.residual_block(input_data, 512, 256, 512)

        route_2 = input_data
        input_data = self.convolutional(input_data, (3, 3, 512, 1024), downsample=True)

        for i in range(4):
            input_data = self.residual_block(input_data, 1024, 512, 1024)

        return route_1, route_2, input_data
    
    
    def yolov3(self,input_layer, NUM_CLASS):
        # After the input layer enters the Darknet-53 network, we get three branches
        route_1, route_2, conv = self.darknet53(input_layer)
        # See the orange module (DBL) in the figure above, a total of 5 Subconvolution operation
        conv = self.convolutional(conv, (1, 1, 1024,  512))
        conv = self.convolutional(conv, (3, 3,  512, 1024))
        conv = self.convolutional(conv, (1, 1, 1024,  512))
        conv = self.convolutional(conv, (3, 3,  512, 1024))
        conv = self.convolutional(conv, (1, 1, 1024,  512))
        conv_lobj_branch = self.convolutional(conv, (3, 3, 512, 1024))

        # conv_lbbox is used to predict large-sized objects , Shape = [None, 13, 13, 255] 
        conv_lbbox = self.convolutional(conv_lobj_branch, (1, 1, 1024, 3*(NUM_CLASS + 5)), activate=False, bn=False)

        conv = self.convolutional(conv, (1, 1,  512,  256))
        # upsample here uses the nearest neighbor interpolation method, which has the advantage that the
        # upsampling process does not need to learn, thereby reducing the network parameter  
        conv = self.upsample(conv)

        conv = tf.concat([conv, route_2], axis=-1)
        conv = self.convolutional(conv, (1, 1, 768, 256))
        conv = self.convolutional(conv, (3, 3, 256, 512))
        conv = self.convolutional(conv, (1, 1, 512, 256))
        conv = self.convolutional(conv, (3, 3, 256, 512))
        conv = self.convolutional(conv, (1, 1, 512, 256))
        conv_mobj_branch = self.convolutional(conv, (3, 3, 256, 512))

        # conv_mbbox is used to predict medium-sized objects, shape = [None, 26, 26, 255]
        conv_mbbox = self.convolutional(conv_mobj_branch, (1, 1, 512, 3*(NUM_CLASS + 5)), activate=False, bn=False)

        conv = self.convolutional(conv, (1, 1, 256, 128))
        conv = self.upsample(conv)

        conv = tf.concat([conv, route_1], axis=-1)
        conv = self.convolutional(conv, (1, 1, 384, 128))
        conv = self.convolutional(conv, (3, 3, 128, 256))
        conv = self.convolutional(conv, (1, 1, 256, 128))
        conv = self.convolutional(conv, (3, 3, 128, 256))
        conv = self.convolutional(conv, (1, 1, 256, 128))
        conv_sobj_branch = self.convolutional(conv, (3, 3, 128, 256))

        # conv_sbbox is used to predict small size objects, shape = [None, 52, 52, 255]
        conv_sbbox = self.convolutional(conv_sobj_branch, (1, 1, 256, 3*(NUM_CLASS +5)), activate=False, bn=False)

        return [conv_sbbox, conv_mbbox, conv_lbbox]
    
    def yolo_micro(self,input_data):
        input_data = self.convolutional(input_data, (3, 3, 3, self.N))
        input_data = MaxPool2D(2, 2, 'same')(input_data)
        input_data = self.convolutional(input_data, (3, 3, self.N, 2*self.N))
        input_data = MaxPool2D(2, 2, 'same')(input_data)
        input_data = self.convolutional(input_data, (3, 3, 2*self.N, 4*self.N))
        input_data = MaxPool2D(2, 2, 'same')(input_data)
        input_data = self.convolutional(input_data, (3, 3, 4*self.N, 8*self.N))
        input_data = MaxPool2D(2, 2, 'same')(input_data)
        input_data = self.convolutional(input_data, (3, 3, 8*self.N, 16*self.N))
        route_1 = input_data
        input_data = MaxPool2D(2, 2, 'same')(input_data)
        input_data = self.convolutional(input_data, (3, 3, 16*self.N, 32*self.N))
        input_data = MaxPool2D(2, 1, 'same')(input_data)
        input_data = self.convolutional(input_data, (3, 3, 32*self.N, 64*self.N))

        return route_1, input_data
    
    def yolov3_micro(self,input_layer, NUM_CLASS):
        # After the input layer enters the Darknet-19 network, we get 2 branches
        route_1, conv = self.yolo_micro(input_layer)

        conv = self.convolutional(conv, (1, 1, 64*self.N, 16*self.N))
        conv_lobj_branch = self.convolutional(conv, (3, 3, 16*self.N, 32*self.N))

        # conv_lbbox is used to predict large-sized objects , Shape = [None, 26, 26, 255]
        conv_lbbox = self.convolutional(conv_lobj_branch, (1, 1, 32*self.N, 3*(NUM_CLASS + 5)), activate=False, bn=False)

        conv = self.convolutional(conv, (1, 1, 16*self.N, 8*self.N))
        # upsample here uses the nearest neighbor interpolation method, which has the advantage that the
        # upsampling process does not need to learn, thereby reducing the network parameter  
        conv = self.upsample(conv)

        conv = tf.concat([conv, route_1], axis=-1)
        conv_mobj_branch = self.convolutional(conv, (3, 3, 8*self.N, 16*self.N))
        # conv_mbbox is used to predict medium size objects, shape = [None, 13, 13, 255]
        conv_mbbox = self.convolutional(conv_mobj_branch, (1, 1, 16*self.N, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        return [conv_mbbox, conv_lbbox]
    
    def conv_ups_block(self,y1,y2,d,n=2,_act=True):
                
        y1 = Conv2D(d,(1,1))(y1)
        y1 = UpSampling2D(size=(n,n))(y1)
        y2 = Conv2D(d,(1,1))(y2)
        y = Add()([y1,y2])
        if _act:
            y = BatchNormalization()(y)
            y = LeakyReLU(alpha=0.1)(y)
        #y = UpSampling2D(size=(n,n))(y)
        return y
    
    def Ups2D(self,x):
        x = Conv2DTranspose(self.seg_out_shape, 1,strides=(2,2), activation = 'relu', padding = 'same')(x)
        return x
    
    def yolov3_lite(self,input_data):
        se_block = TRAIN_USE_SE_LAYERS
        x1 = self.convolutional(input_data, (3, 3, 3, self.N))
        if se_block[0]: 
            x1 = self.squeeze_excite_block(x1)
            x1 = self.spatial_attention_block(x1)
        route_1 = x1
        
        x1 = MaxPool2D(2, 2, 'same')(x1)
    
        x2 = self.convolutional(x1, (3, 3, self.N, 2*self.N))
        if se_block[1]: 
            x2 = self.squeeze_excite_block(x2)
            x2 = self.spatial_attention_block(x2)
        route_2 = x2
        
        
        x2 = MaxPool2D(2, 2, 'same')(x2)
        
        x3 = self.convolutional(x2, (3, 3, 2*self.N, 4*self.N))
        if se_block[2]: 
            x3 = self.squeeze_excite_block(x3)
            x3 = self.spatial_attention_block(x3)
        route_3 = x3
        
        x3 = MaxPool2D(2, 2, 'same')(x3)
        
        x4 = self.convolutional(x3, (3, 3, 4*self.N, 8*self.N))
        if se_block[3]: 
            x4 = self.squeeze_excite_block(x4)
            x4 = self.spatial_attention_block(x4)
        route_4 = x4
        
        x4 = MaxPool2D(2, 2, 'same')(x4)
        
        x5 = self.convolutional(x4, (3, 3, 8*self.N, 8*self.N))
        if se_block[4]: 
            x5 = self.squeeze_excite_block(x5)
            x5 = self.spatial_attention_block(x5)
        route_5 = x5
        
        x5 = MaxPool2D(2, 2, 'same')(x5)
        
        x6 = self.convolutional(x5, (3, 3, 8*self.N, 8*self.N))
        if se_block[5]: 
            x6 = self.squeeze_excite_block(x6)
            x6 = self.spatial_attention_block(x6)
        
        x6 = MaxPool2D(2, 1, 'same')(x6)
        
        x7 = self.convolutional(x6, (3, 3, 8*self.N, 16*self.N))
        x8 = self.convolutional(x7, (1, 1, 16*self.N, 8*self.N))
        route_6 = x8
        
        fpn=[]
        self.seg_out_shape = len(CLASS_NAMES) + 1
        
        if TRAIN_USE_DST:           
            _route_2 = Conv2D(1,(1,1))(route_2)
            _route_3 = Conv2D(1,(1,1))(route_3)
            _route_4 = Conv2D(1,(1,1))(route_4)
            
            y = UpSampling2D(size=(2,2))(_route_4)
            y = Add()([y,_route_3])
            
            y = UpSampling2D(size=(2,2))(y)
            y = Add()([y,_route_2])
            
            y = ReLU()(y)
            fpn+= [y]
            
        if TRAIN_USE_SEG:
            
            y1 = self.conv_ups_block(route_6,route_5,self.seg_out_shape,n=2)
            y2 = self.conv_ups_block(y1,route_4,self.seg_out_shape,n=2)
            y3 = self.conv_ups_block(y2,route_3,self.seg_out_shape,n=2)
            y4 = self.conv_ups_block(y3,route_2,self.seg_out_shape,n=2,_act=False)
            #y5 = UpSampling2D(size=(2,2))(y4)
            y  = Sigmoid(y5)
            fpn += [y]
            
        
        conv_lobj_branch = self.convolutional(route_6, (3, 3, 8*self.N, 16*self.N))
        conv_lbbox = self.convolutional(conv_lobj_branch, (1, 1, 16*self.N, 3*(NUM_CLASS + 5)), activate=False, bn=False)

        conv = self.convolutional(route_6, (1, 1, 8*self.N, 4*self.N))
        conv = self.upsample(conv)
        
        conv = tf.concat([conv, route_5], axis=-1)
        conv_mobj_branch = self.convolutional(conv, (3, 3, 8*self.N, 16*self.N))
        conv_mbbox = self.convolutional(conv_mobj_branch, (1, 1, 16*self.N, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        return [conv_mbbox, conv_lbbox]+fpn 
    
    
    def yolov3_tiny_core(self,input_data):
        input_data = self.convolutional(input_data, (3, 3, 3, 16))
        input_data = MaxPool2D(2, 2, 'same')(input_data)
        input_data = self.convolutional(input_data, (3, 3, 16, 32))
        input_data = MaxPool2D(2, 2, 'same')(input_data)
        input_data = self.convolutional(input_data, (3, 3, 32, 64))
        input_data = MaxPool2D(2, 2, 'same')(input_data)
        input_data = self.convolutional(input_data, (3, 3, 64, 128))
        input_data = MaxPool2D(2, 2, 'same')(input_data)
        input_data = self.convolutional(input_data, (3, 3, 128, 256))
        route_1 = input_data
        input_data = MaxPool2D(2, 2, 'same')(input_data)
        input_data = self.convolutional(input_data, (3, 3, 256, 512))
        input_data = MaxPool2D(2, 1, 'same')(input_data)
        input_data = self.convolutional(input_data, (3, 3, 512, 1024))

        return route_1, input_data
    
    
    def yolov3_tiny(self,input_layer, NUM_CLASS):
        # After the input layer enters the Darknet-19 network, we get 2 branches
        route_1, conv = self.yolov3_tiny_core(input_layer)

        conv = self.convolutional(conv, (1, 1, 1024, 256))
        conv_lobj_branch = self.convolutional(conv, (3, 3, 256, 512))

        # conv_lbbox is used to predict large-sized objects , Shape = [None, 26, 26, 255]
        conv_lbbox = self.convolutional(conv_lobj_branch, (1, 1, 512, 3*(NUM_CLASS + 5)), activate=False, bn=False)

        conv = self.convolutional(conv, (1, 1, 256, 128))
        # upsample here uses the nearest neighbor interpolation method, which has the advantage that the
        # upsampling process does not need to learn, thereby reducing the network parameter  
        conv = self.upsample(conv)

        conv = tf.concat([conv, route_1], axis=-1)
        conv_mobj_branch = self.convolutional(conv, (3, 3, 128, 256))
        # conv_mbbox is used to predict medium size objects, shape = [None, 13, 13, 255]
        conv_mbbox = self.convolutional(conv_mobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        return [conv_mbbox, conv_lbbox]
    
    def mobilenet_v2(self):
        from .mobilenet_v2 import MobileNetV2
        if NO_OF_GRID == 2 : OBJ_SCALES = ['medium','large']
        if NO_OF_GRID == 3 : OBJ_SCALES = ['small','medium','large']
        
        self.yolo_model = MobileNetV2(input_shape=(TRAIN_INPUT_SIZE,TRAIN_INPUT_SIZE,3),
                    input_tensor=None,
                    alpha=1.0,
                    pooling=None,
                    NUM_CLASS=NUM_CLASS,
                    N  = int(TRAIN_MODEL_SCALE),
                    NH = int(TRAIN_MODEL_HEAD_SCALE), 
                    OBJ_SCALES = OBJ_SCALES)
        self.yolo_model.build(input_shape=(None,TRAIN_INPUT_SIZE,TRAIN_INPUT_SIZE,3))
        if len(YOLO_MODEL_LOAD_WTS) : self.yolo_model.set_wts(YOLO_MODEL_LOAD_WTS)
    
    def shufflenet_v2(self):
        from applications.shufflenet import ShuffleNetV2
        conv_out,out = ShuffleNetV2(include_top=False,
                    input_tensor=input_tensor,
                    scale_factor=1.0,
                    pooling='max',
                    input_shape=None,
                    load_model=None,
                    num_shuffle_units=[3,7,3],
                    bottleneck_ratio=1,
                    classes=1000)
        
        return conv_out,out
    
    #@tf.function
    def decode_output(self,conv_tensors,batch_size=TRAIN_MINI_BATCH_SIZE):
        output_tensors = []
        for i, conv_output in enumerate(conv_tensors[:NO_OF_GRID]):
            # where i = 0, 1 or 2 to correspond to the three grid scales  
            #conv_shape       = TRAIN_MINI_BATCH_SIZE,GRIDS[i]#tf.shape(conv_output)
            #batch_size      = conv_shape[0]
            output_size      = GRIDS[i]

            conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

            conv_raw_dxdy = conv_output[:, :, :, :, 0:2] # offset of center position     
            conv_raw_dwdh = conv_output[:, :, :, :, 2:4] # Prediction box length and width offset
            conv_raw_conf = conv_output[:, :, :, :, 4:5] # confidence of the prediction box
            conv_raw_prob = conv_output[:, :, :, :, 5: ] # category probability of the prediction box 

            # next need Draw the grid. Where output_size is equal to 13, 26 or 52  
            y = tf.range(output_size, dtype=tf.int32)
            y = tf.expand_dims(y, -1)
            y = tf.tile(y, [1, output_size])
            x = tf.range(output_size,dtype=tf.int32)
            x = tf.expand_dims(x, 0)
            x = tf.tile(x, [output_size, 1])

            xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
            xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
            xy_grid = tf.cast(xy_grid, tf.float32)

            #https://github.com/ultralytics/yolov5/issues/471
            if LOSS_XYWH_IOU and LOSS_GRID_CORR :
                pred_xy = 2*tf.sigmoid(conv_raw_dxdy) -0.5
                pred_xy = (pred_xy + xy_grid) / tf.cast(output_size,tf.float32)
                if LOSS_WH_POW > 0: 
                    pred_wh = tf.pow(2*tf.sigmoid(conv_raw_dwdh),LOSS_WH_POW) * ANCHORS[i]
                else: 
                    pred_wh = tf.exp(conv_raw_dwdh) * ANCHORS[i]

            else :
                pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) / tf.cast(output_size,tf.float32)
                pred_wh = tf.exp(conv_raw_dwdh) * ANCHORS[i]

            pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
            pred_conf = tf.sigmoid(conv_raw_conf) 
            pred_prob = tf.sigmoid(conv_raw_prob) 
            decoded_output = tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)
            if self._training: output_tensors.append(conv_output)
            output_tensors.append(decoded_output)
            
        return output_tensors
    
    def load_yolo_weights(self,model, weights_file):
        tf.keras.backend.clear_session() # used to reset layer names
        # load Darknet original weights to TensorFlow model
        if YOLO_TYPE == "yolov3":
            range1 = 75 if not TRAIN_YOLO_TINY else 13
            range2 = [58, 66, 74] if not TRAIN_YOLO_TINY else [9, 12]
        if YOLO_TYPE == "yolov4":
            range1 = 110 if not TRAIN_YOLO_TINY else 21
            range2 = [93, 101, 109] if not TRAIN_YOLO_TINY else [17, 20]

        with open(weights_file, 'rb') as wf:
            major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

            j = 0
            for i in range(range1):
                if i > 0:
                    conv_layer_name = 'conv2d_%d' %i
                else:
                    conv_layer_name = 'conv2d'

                if j > 0:
                    bn_layer_name = 'batch_normalization_%d' %j
                else:
                    bn_layer_name = 'batch_normalization'
                conv_layer_name = 'conv_%d' %i
                
                conv_layer = model.get_layer(conv_layer_name)
                filters = conv_layer.filters
                k_size = conv_layer.kernel_size[0]
                in_dim = conv_layer.input_shape[-1]

                if i not in range2:
                    # darknet weights: [beta, gamma, mean, variance]
                    bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                    # tf weights: [gamma, beta, mean, variance]
                    bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                    bn_layer = model.get_layer(bn_layer_name)
                    j += 1
                else:
                    conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

                # darknet shape (out_dim, in_dim, height, width)
                conv_shape = (filters, in_dim, k_size, k_size)
                conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
                # tf shape (height, width, in_dim, out_dim)
                conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

                if i not in range2:
                    conv_layer.set_weights([conv_weights])
                    bn_layer.set_weights(bn_weights)
                else:
                    conv_layer.set_weights([conv_weights, conv_bias])

            assert len(wf.read()) == 0, 'failed to read all data'


def bbox_iou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = 1.0 * inter_area / union_area
    iou = tf.maximum(iou,np.finfo(np.float32).eps)
    return iou

def bbox_giou(boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    # Calculate the iou value between the two bounding boxes
    iou = inter_area / union_area

    # Calculate the coordinates of the upper left corner and the lower right corner of the smallest closed convex surface
    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

    # Calculate the area of the smallest closed convex surface C
    enclose_area = enclose[..., 0] * enclose[..., 1]

    # Calculate the GIoU value according to the GioU formula  
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou

# testing (should be better than giou)
def bbox_iou_loss(boxes1, boxes2,eps=1e-7):
    iou = bbox_iou(boxes1, boxes2)
    if LOSS_IOU_TYPE == 'iou' : return iou

    boxes1_coor = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2_coor = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
    
    ar_1 = boxes1[...,2]/(boxes1[...,3] + eps)
    ar_2 = boxes2[...,2]/(boxes2[...,3] + eps)
    
    left_up    = tf.minimum(boxes1_coor[..., :2], boxes2_coor[..., :2])
    right_down = tf.maximum(boxes1_coor[..., 2:], boxes2_coor[..., 2:])
    
    cxy_1  = boxes1_coor[..., :2] + 0.5* boxes1_coor[..., 2:]
    cxy_2  = boxes2_coor[..., :2] + 0.5* boxes2_coor[..., 2:]

    dia_dist = tf.reduce_sum(tf.square(left_up - right_down),axis=-1)
    cnt_dist = tf.reduce_sum(tf.square(cxy_1 - cxy_2),axis=-1) + eps
    
    v = (4 / np.pi ** 2) * tf.pow(tf.math.atan(ar_2) - tf.math.atan(ar_1), 2)
    alpha = v / (v - iou + (1 + eps))
    
    if LOSS_IOU_TYPE == 'ciou' : iou =  iou - ((cnt_dist/dia_dist) + (v*alpha))
    if LOSS_IOU_TYPE == 'diou' : iou =  iou - (cnt_dist/dia_dist) 
    #carea = (left_up - right_down)[...,0] * (left_up - right_down)[...,1]
    #giou = iou - (carea -
    return iou

#@tf.function
def calc_yolo_loss(pred, conv, label, bboxes, i=0, batch_size = TRAIN_MINI_BATCH_SIZE):
    #https://github.com/ultralytics/yolov5/issues/6998
    #conv_shape  = tf.shape(conv)
    #batch_size  = TRAIN_MINI_BATCH_SIZE#conv_shape[0]
    output_size = GRIDS[i]#conv_shape[1]
    
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh     = pred[:, :, :, :, 0:4]
    pred_conf     = pred[:, :, :, :, 4:5]

    label_xywh    = label[:, :, :, :, 0:4]
    label_bbox    = label[:, :, :, :, 4:5]
    label_prob    = label[:, :, :, :, 5:]
    
    if LOSS_XYWH_MSE:
        pred_xywh[...,2:]  = pred_xywh[...,2:]/ANCHORS[i]
        pred_xywh[...,2:]  = tf.log(pred_xywh[...,2:])
        pred_xywh[...,2:]  = tf.sign(pred_xywh[...,2:]) * tf.sqrt(tf.abs(pred_xywh[...,2:]))
        
        label_xywh[...,2:] = label_xywh[...,2:]/ANCHORS[i]
        label_xywh[...,2:] = tf.log(label_xywh[...,2:])
        label_xywh[...,2:] = tf.sign(label_xywh[...,2:]) * tf.sqrt(tf.abs(label_xywh[...,2:]))
        
        y = tf.range(output_size, dtype=tf.int32)
        y = tf.expand_dims(y, -1)
        y = tf.tile(y, [1, output_size])
        x = tf.range(output_size,dtype=tf.int32)
        x = tf.expand_dims(x, 0)
        x = tf.tile(x, [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xywh[...,:2]  = pred_xywh[...,:2]  * tf.cast(output_size,tf.float32) - xy_grid
        label_xywh[...,:2] = label_xywh[...,:2] * tf.cast(output_size,tf.float32) - xy_grid
        
        xy_loss   = 5.0 * tf.square(pred_xywh[...,:2] - label_xywh[...,:2])
        wh_loss   = 5.0 * tf.square(pred_xywh[...,2:] - label_xywh[...,2:])
        bbox_loss = xy_loss + wh_loss
        
    if LOSS_XYWH_IOU:  
        bbox_loss = tf.expand_dims(bbox_iou_loss(pred_xywh, label_xywh), axis=-1)
    
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] #/ (input_size ** 2)
    bbox_loss = label_bbox * bbox_loss_scale * (1 - bbox_loss)
    
    bg_mask = 1.0
    if LOSS_FILTER_BG_MASK:
        iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
        bg_mask = tf.cast( max_iou < LOSS_BG_IOU_THRESH, tf.float32 )
    label_bgd = (1.0 - label_bbox) * bg_mask
    
    conf_focal=1.0
    if LOSS_USE_FOCAL:
        conf_focal  = tf.pow(label_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
            label_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_bbox, logits=conv_raw_conf)
            +
            label_bgd  * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_bbox, logits=conv_raw_conf)
    )

    prob_loss = label_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    bbox_loss = tf.math.reduce_sum(bbox_loss, axis=[1,2,3,4])
    conf_loss = tf.math.reduce_sum(conf_loss, axis=[1,2,3,4])
    prob_loss = tf.math.reduce_sum(prob_loss, axis=[1,2,3,4])
    
    bbox_loss = tf.cast(LOSS_WTS_BBOX[i],tf.float32) * tf.math.reduce_mean(bbox_loss)
    conf_loss = tf.math.reduce_mean(conf_loss)
    prob_loss = tf.math.reduce_mean(prob_loss)
    
    _loss_dict={}
    _loss_dict['iou_loss']  = bbox_loss
    _loss_dict['conf_loss'] = conf_loss 
    _loss_dict['prob_loss'] = prob_loss
    _loss_dict['det_loss']  = MTL_LOSS_WTS[0] * bbox_loss + MTL_LOSS_WTS[1] * conf_loss + MTL_LOSS_WTS[2] * prob_loss
    
    return _loss_dict, bbox_loss , conf_loss , prob_loss

#@tf.function
def calc_yolo_loss_v3(pred, conv, label, bboxes, i=0):
    conv_shape  = tf.shape(conv)
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh     = pred[:, :, :, :, 0:4]
    pred_conf     = pred[:, :, :, :, 4:5]
    
    label_xywh    = label[:, :, :, :, 0:4]
    label_bbox    = label[:, :, :, :, 4:5]
    label_prob    = label[:, :, :, :, 5:]
    
    
    pred_xywh[...,2:]  = (pred_xywh[...,2:]*output_size)/ANCHORS[i]
    pred_xywh[...,2:]  = tf.clip_by_value(pred_xywh[...,2:],1.0,1e10)
    pred_xywh[...,2:]  = tf.log(pred_xywh[...,2:] + zero_mask)
    
    label_xywh[...,2:] = (label_xywh[...,2:]*output_size)/ANCHORS[i]
    label_xywh[...,2:] = tf.clip_by_value(label_xywh[...:2],1.0,1e10)
    label_xywh[...,2:] = tf.log(label_xywh[...,2:] + zero_mask)
    
    xy_loss = 5.0 * tf.square(pred_xywh[...,:2] - label_xywh[...,:2])
    wh_loss = 5.0 * tf.square(tf.sqrt(pred_xywh[...,2:]) - tf.sqrt(label_xywh[...,2:]))
    
    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
    
    label_bgd = (1.0 - label_bbox) * tf.cast( max_iou < YOLO_IOU_LOSS_THRESH, tf.float32 )
    #conf_focal  = tf.pow(label_bbox - pred_conf, 2)

    conf_loss = (
            1.0 * label_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_bbox, logits=conv_raw_conf)
            +
            0.5 * label_bgd  * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_bbox, logits=conv_raw_conf)
    )

    prob_loss = label_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)
    
    bbox_sum = tf.math.reduce_sum(xy_loss, axis=[1,2,3,4]) + tf.math.reduce_sum(wh_loss, axis=[1,2,3,4])
    conf_sum = tf.math.reduce_sum(conf_loss, axis=[1,2,3,4])
    prob_sum = tf.math.reduce_sum(prob_loss, axis=[1,2,3,4])
    total_loss_sum = giou_sum + conf_sum + prob_sum
    
    bbox_loss = tf.math.reduce_mean(bbox_sum)
    conf_loss = tf.math.reduce_mean(conf_sum)
    prob_loss = tf.math.reduce_mean(prob_sum)
    
    total_loss = bbox_loss + conf_loss + prob_loss
    
    _loss_dict={}; 
    _loss_dict['iou_loss']  = bbox_loss
    _loss_dict['conf_loss'] = conf_loss
    _loss_dict['prob_loss'] = prob_loss
    _loss_dict['det_loss']  = total_loss
    
    return _loss_dict, bbox_loss , conf_loss , prob_loss


'''
def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    Return yolo_loss tensor
    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss
    Returns
    -------
    loss: tensor, shape=(1,)
    
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
    return loss
'''


#@tf.function
def calc_seg_loss(_labels,_preds):
    _loss_dict={}; _smp_loss_dict={}
    avg_seg_loss = 0; avg_dst_loss =0; dst_loss =0 ; seg_loss =0;
    
    if TRAIN_USE_DST:  
        _label = _labels[0][:,:,:,tf.newaxis]
        _pred = _preds[0]
        #mse  = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        fg_mask = tf.cast(_label>0,dtype=tf.float32)
        bg_mask = 1.0 - fg_mask
        mse_loss_fg = fg_mask*tf.square(_label-_pred)
        mse_loss_bg = (1-fg_mask)*tf.square(_label-_pred)
        
        dst_loss = tf.math.reduce_sum(mse_loss_fg,axis=[1,2])/tf.math.reduce_sum(fg_mask)\
                      + tf.math.reduce_sum(mse_loss_bg,axis=[1,2])/tf.math.reduce_sum(bg_mask)
        dst_loss = tf.math.sqrt(dst_loss)
        #tf.math.log    = tf.math.log(rmse_loss_mean)
        
        dst_std  = tf.math.reduce_std(dst_loss)
        avg_dst_loss = tf.math.reduce_mean(dst_loss)
        _loss_dict['dst_loss']= avg_dst_loss
        _loss_dict['dst_std']=dst_std
        _smp_loss_dict['dst'] = dst_loss  
    
    if TRAIN_USE_SEG:     
        seg_label = tf.one_hot(_labels[1],80)
        #bce  = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        #bce_loss = bce(seg_label,seg_pred)
        _min = 10e-8
        seg_loss = - seg_label     * tf.math.log(tf.clip_by_value(seg_pred+_min,0,1))\
                   - (1-seg_label) * tf.math.log(tf.clip_by_value(1-seg_pred+_min,0,1))
        seg_loss = tf.math.reduce_mean(seg_loss,axis=[1,2,3])
        
        seg_std  = tf.math.reduce_std(seg_loss)
        avg_seg_loss = tf.math.reduce_mean(seg_loss)
        
        _loss_dict['seg_loss']= avg_seg_loss
        _loss_dict['seg_std']= seg_std
        _smp_loss_dict['seg'] = seg_loss
    
    total_loss = TRAIN_LOSS_WTS[1] * avg_dst_loss + TRAIN_LOSS_WTS[2] * avg_seg_loss
    total_smp_loss = TRAIN_LOSS_WTS[1] * dst_loss + TRAIN_LOSS_WTS[2] * seg_loss
    return _loss_dict,_smp_loss_dict, total_smp_loss, total_loss
    
    