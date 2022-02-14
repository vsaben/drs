"""
    Description: Yolo backbones and their supporting functions
    Function: Build Yolo graph
    - PART A: Individual layers and blocks
    - PART B: Darknet
    - PART C: Yolo head supporting functions
    - PART D: Yolo
    - PART E: Choose yolo backbone
    - PART F: Freeze darknet layers
"""

import tensorflow as tf
import tensorflow.keras as K 
import tensorflow.keras.layers as KL 

from methods.utils import CenterToMinMax, GridToImg

# PART A: Individual layers and blocks ================================================

def DarknetConv(x, filters, kernel, strides=1, batch_norm=True, activate_type = 'leaky'):
    if strides == 1:
        padding = 'same'
    else:
        x = KL.ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = KL.Conv2D(filters=filters, kernel_size=kernel,
                  strides=strides, padding=padding,
                  use_bias=not batch_norm)(x)             
    if batch_norm:
        x = KL.BatchNormalization()(x) 
        if activate_type == 'leaky': x = KL.LeakyReLU(alpha=0.1)(x)
        if activate_type == 'mish': x = mish(x)
    return x

def mish(x):
    return KL.Lambda(lambda x: x * tf.math.tanh(tf.math.softplus(x)))(x)

class BatchNorm(KL.Layer):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bn = KL.BatchNormalization(momentum=0.9)

    def call(self, x, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """

        return self.bn(x) #super(self.__class__, self).call(inputs, training=training)

def DarknetResidual(x, filters, activate_type = 'leaky'): 
    prev = x
    x = DarknetConv(x, filters // 2, 1, activate_type = activate_type)
    x = DarknetConv(x, filters, 3, activate_type = activate_type)
    x = KL.Add()([prev, x])
    return x

def CSPDarknetResidual(x, filters, activate_type = 'mish'):
    prev = x
    x = DarknetConv(x, filters, 1, activate_type = activate_type)
    x = DarknetConv(x, filters, 3, activate_type = activate_type)
    x = KL.Add()([prev, x])
    return x

def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x

def CSPDarknetBlock(x, filters, blocks, residual_type = 'yolo4'):
    rt = DarknetConv(x, filters, 1, activate_type = 'mish')
    x = DarknetConv(x, filters, 1, activate_type= 'mish')

    DarknetResidualFunc = CSPDarknetResidual if residual_type == 'yolo4' else DarknetResidual
    for _ in range(blocks):
        x = DarknetResidualFunc(x, filters, activate_type = 'mish')
    x = DarknetConv(x, filters, 1, activate_type= 'mish')
    
    x = KL.Add()([x, rt])
    return x

def CSPDarknetBlockTiny(x, filters, return_rt_last = False):
    rt0 = x
    x = route_group(x, 2, 1)
    x = rt1 = DarknetConv(x, filters, 3)
    x = DarknetConv(x, filters, 3)
    x = KL.Add()([x, rt1])
    x = rt_last = DarknetConv(x, 2*filters, 1)
    x = KL.Add()([rt0, x])
    x = KL.MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 4*filters, 3)
    if return_rt_last: return x, rt_last
    return x

def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]

def CSPMaxPool(kernel):
    return KL.Lambda(lambda x: tf.nn.max_pool(x, kernel, strides=1, padding='SAME'))



# PART B: Darknet =====================================================================

def Darknet(x, name=None):
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  
    x = x_36 = DarknetBlock(x, 256, 8)  
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return x_36, x_61, x 

def CSPDarknet(x, name=None):
    x = DarknetConv(x, 32, 3, activate_type = 'mish')
    x = DarknetConv(x, 64, 3, 2, activate_type = 'mish')    
    x = CSPDarknetBlock(x, 64, 1, residual_type = 'yolo3')
    x = DarknetConv(x, 64, 1, activate_type = 'mish')
    x = DarknetConv(x, 128, 3, 2, activate_type = 'mish') 
    x = CSPDarknetBlock(x, 64, 2)
    x = DarknetConv(x, 128, 1, activate_type = 'mish')
    x = DarknetConv(x, 256, 3, 2, activate_type = 'mish') 
    x = CSPDarknetBlock(x, 128, 8)
    x = x_1 = DarknetConv(x, 256, 1, activate_type = 'mish')
    x = DarknetConv(x, 512, 3, 2, activate_type = 'mish')
    x = CSPDarknetBlock(x, 256, 8)
    x = x_2 = DarknetConv(x, 512, 1, activate_type = 'mish')
    x = DarknetConv(x, 1024, 3, 2, activate_type = 'mish')
    x = CSPDarknetBlock(x, 512, 4)
    
    x = DarknetConv(x, 1024, 1, activate_type = 'mish')
    x = DarknetConv(x, 512, 1)
    x = DarknetConv(x, 1024, 3)
    x = DarknetConv(x, 512, 1)

    x = KL.Add()([CSPMaxPool(13)(x), CSPMaxPool(9)(x), CSPMaxPool(5)(x), x])
    x = DarknetConv(x, 512, 1)
    x = DarknetConv(x, 1024, 3)
    x = DarknetConv(x, 512, 1)
    return x_1, x_2, x

def DarknetTiny(x, name=None):
    x = DarknetConv(x, 16, 3)
    x = KL.MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 32, 3)
    x = KL.MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 64, 3)
    x = KL.MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 128, 3)
    x = KL.MaxPool2D(2, 2, 'same')(x)
    x = x_8 = DarknetConv(x, 256, 3)  
    x = KL.MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 512, 3)
    x = KL.MaxPool2D(2, 1, 'same')(x)
    x = DarknetConv(x, 1024, 3)
    return x_8, x

def CSPDarknetTiny(x, name=None):
    x = DarknetConv(x, 32, 3, 2)
    x = DarknetConv(x, 64, 3, 2)
    x = DarknetConv(x, 64, 3)
    x = CSPDarknetBlockTiny(x, 32)
    x = CSPDarknetBlockTiny(x, 64)
    x, rt1 = CSPDarknetBlockTiny(x, 128, return_rt_last = True)
    return rt1, x

# PART C: Yolo head supporting functions ==================================================

def YoloConv(x_in, filters, iscsp=False, name=None): 
    if isinstance(x_in, tuple):
        x, x_skip = x_in                       
        x = DarknetConv(x, filters, 1)
        if iscsp: x_skip = DarknetConv(x_skip, filters, 1)
        x = KL.UpSampling2D(2)(x)
        x = KL.Concatenate()([x, x_skip])
    else:
        x = x_in                       

    x = DarknetConv(x, filters, 1)
    x = DarknetConv(x, filters * 2, 3)
    x = DarknetConv(x, filters, 1)
    x = DarknetConv(x, filters * 2, 3)
    x = DarknetConv(x, filters, 1)

    return x

def CSPYoloConv(x_in, filters, name=None): 
    x, rt = x_in
    x = DarknetConv(x, filters, 3, 2)
    x = KL.Concatenate()([x, rt])
    x = YoloConv(x, filters)
    return x

def YoloConvTiny(x_in, filters, name=None): 
    if isinstance(x_in, tuple):
        x, x_skip = x_in
        x = DarknetConv(x, filters, 1)
        x = KL.UpSampling2D(2)(x)
        x = KL.Concatenate()([x, x_skip])
    else:
        x = x_in 
        x = DarknetConv(x, filters, 1)
    return x

def YoloOutput(x, filters, nanchors, classes = 2, name=None):   
    x = DarknetConv(x, filters * 2, 3)
    x = DarknetConv(x, nanchors * (classes + 5), 1, batch_norm=False)
    x = KL.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], nanchors, classes + 5)))(x)
    return x

def Boxes(conv_output, anchors_i, xyscale_i, classes = 2):

    """Extract bounding boxes, objectness scores and class probabilities from 
    yolo predictions
    :note: adapt undamaged/damaged class output into a probability of damage
    
    :param pred: (nbatch, grid, grid, (x, y, w, h, P(damage)))
    :param anchors: scale-specific bounding box anchors
    :param classes: number of class predictions (refer note)

    :result bbox_xyxy: [nbatch, grid, grid, 3, 4] (normalised)
    :result obj: [nbatch, grid, grid, 3, 1]
    :result class_probs: [nbatch, grid, grid, 3, nclass]
    :result pred_box_xywh: [nbatch, grid, grid, 3, 4] (original)
    """

    output_size = tf.shape(conv_output)[1]
    conv_raw_dxdy, conv_raw_dwdh, conv_raw_obj, conv_raw_class = tf.split(conv_output, (2, 2, 1, classes), axis=-1) 

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2) 
    xy_grid = tf.cast(xy_grid, tf.float32)

    conv_xy = tf.sigmoid(conv_raw_dxdy)
    pred_box = tf.concat([conv_xy, conv_raw_dwdh], axis=-1) # for loss

    pred_xy = ((conv_xy * xyscale_i) - 0.5 * (xyscale_i - 1) + xy_grid) / tf.cast(output_size, tf.float32)
    pred_wh = tf.exp(conv_raw_dwdh) * anchors_i 

    pred_x1y1 = pred_xy - pred_wh / 2
    pred_x2y2 = pred_xy + pred_wh / 2
    pred_bbox = tf.concat([pred_x1y1, pred_x2y2], axis=-1)

    pred_obj = tf.sigmoid(conv_raw_obj)
    pred_class = tf.sigmoid(conv_raw_class)  

    return pred_bbox, pred_obj, pred_class, pred_box

# PART D: Yolo =========================================================================

def YoloV3(x, cfg):

    masks = cfg['MASKS']
    classes = cfg['NUM_CLASSES']

    x_36, x_61, x = Darknet(x, name = 'darknet')

    x = P0 = YoloConv(x, 512, name='yolo_conv_0')
    out_0  = YoloOutput(x, 512, len(masks[0]), classes, name='yolo_out_0') # LARGE/COARSE GRIDS [13]

    x = P1 = YoloConv((x, x_61), 256, name='yolo_conv_1')
    out_1  = YoloOutput(x, 256, len(masks[1]), classes, name='yolo_out_1') # MEDIUM GRIDS [26]

    x = P2 = YoloConv((x, x_36), 128, name='yolo_conv_2') 
    out_2  = YoloOutput(x, 128, len(masks[2]), classes, name='yolo_out_2') # SMALL/FINE GRIDS [52]

    fmaps  = (P0, P1, P2)
    outs   = (out_0, out_1, out_2) 
   
    return fmaps, outs  
    
def YoloV3T(x, cfg):

    masks = cfg['MASKS']
    classes = cfg['NUM_CLASSES']
    
    x_8, x = DarknetTiny(x, name='darknet')

    x = P0 = YoloConvTiny(x, 256, name='yolo_conv_0')
    out_0  = YoloOutput(x, 256, len(masks[0]), classes, name='yolo_out_0') # LARGE/COARSE GRIDS [13]

    x = P1 = YoloConvTiny((x, x_8), 128, name='yolo_conv_1')
    out_1  = YoloOutput(x, 128, len(masks[1]), classes, name='yolo_out_1') # MEDIUM GRIDS [26]

    fmaps = (P0, P1)
    outs  = (out_0, out_1)

    return fmaps, outs

def YoloV4(x, cfg):
    
    masks = cfg['MASKS']
    classes = cfg['NUM_CLASSES']
    
    rt1, rt2, rt0 = CSPDarknet(x, name='darknet')

    x = rt0
    x = rt2 = YoloConv((x, rt2), 256, name='yolo_conv_00', iscsp = True)

    x = P2 = YoloConv((x, rt1), 128, name='yolo_conv_2', iscsp = True)  
    out_2 = YoloOutput(x, 128, len(masks[2]), classes, name='yolo_out_2') # SMALL/FINE GRIDS [52]
    
    x = P1 = CSPYoloConv((x, rt2), 256, name='yolo_conv_1')
    out_1 = YoloOutput(x, 256, len(masks[1]), classes, name='yolo_out_1') # MEDIUM GRIDS [26]

    x = P0 = CSPYoloConv((x, rt0), 512, name = 'yolo_conv_0')
    out_0 = YoloOutput(x, 512, len(masks[0]), classes, name='yolo_out_0') # LARGE/COARSE GRIDS [13]

    fmaps = (P0, P1, P2)
    outs = (out_0, out_1, out_2)

    return fmaps, outs

def YoloV4T(x, cfg):

    masks = cfg['MASKS']
    classes = cfg['NUM_CLASSES']
    
    rt1, x = CSPDarknetTiny(x, name='darknet')

    x = P0 = YoloConvTiny(x, 256, name='yolo_conv_0')
    out_0 = YoloOutput(x, 256, len(masks[0]), classes, name='yolo_out_0')
  
    x = P1 = YoloConvTiny((x, rt1), 128, name='yolo_conv_1')
    out_1 = YoloOutput(x, 128, len(masks[1]), classes, name='yolo_out_1')

    fmaps = (P0, P1)
    outs = (out_0, out_1)       

    return fmaps, outs

class YoloBox(KL.Layer):

    def __init__(self, cfg, name=None, *args, **kwargs):
        super(YoloBox, self).__init__(name=name, *args, **kwargs)
        self.cfg = cfg
        
        self.anchors = tf.constant(cfg['ANCHORS'], tf.float32)
        self.nlevels = len(cfg['MASKS'])
        self.xyscale = tf.constant(cfg['XYSCALE'], tf.float32)

    def call(self, outs): 
        
        boxes = []
        for i in range(self.nlevels):
            anchors_i = tf.gather(self.anchors, self.cfg['MASKS'][i])
            xyscale_i = self.xyscale[i]
            box = Boxes(outs[i], anchors_i, xyscale_i, self.cfg['NUM_CLASSES']) 
            boxes.append(box)

        return boxes

    def get_config(self):
        config = super(YoloBox, self).get_config()      
        config.update({'cfg': self.cfg})       
        return config
 
class YoloNMS(KL.Layer):
    def __init__(self, mode, cfg, name=None, *args, **kwargs):
        super(YoloNMS, self).__init__(name=name, *args, **kwargs)
        self.mode = mode
        self.cfg = cfg        

    def call(self, inputs):        

        """

        :result boxes: [nbatch, cfg.MAX_GT_INSTANCES, 4] 
        :result scores: [nbatch, cfg.MAX_GT_INSTANCES]
        :result classes: [nbatch, cfg.MAX_GT_INSTANCES]
        :result nvalid: [nbatch]

        :note: zero padding after nvalid
        """

        # Combine and reshape bbox (bbox), objectness / confidence (conf) and class probablities / type (clsp) across grids
        
        bbox = tf.concat([tf.reshape(o[0], (tf.shape(o[0])[0], -1, 4)) for o in inputs], axis=1)
        conf = tf.concat([tf.reshape(o[1], (tf.shape(o[1])[0], -1, 1)) for o in inputs], axis=1)
        clsp = tf.concat([tf.reshape(o[2], (tf.shape(o[2])[0], -1, self.cfg['NUM_CLASSES'])) for o in inputs], axis=1)

        scores = conf * clsp
        bbox = tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4))

        #dscores = tf.squeeze(scores, axis=0)
        #scores = tf.reduce_max(dscores,[1])
        #bbox = tf.reshape(bbox, (-1, 4))     # tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4))
        #classes = tf.argmax(dscores, 1)

        #selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
        #    boxes=bbox,
        #    scores=scores,
        #    max_output_size=self.cfg[MODE + 'MAX_GT_INSTANCES'],
        #    iou_threshold=self.cfg[MODE + 'RPN_IOU_THRESHOLD'],
        #    score_threshold=self.cfg[MODE + 'RPN_SCORE_THRESHOLD'],
        #    soft_nms_sigma=0.5)
    
        #num_valid_nms_boxes = tf.shape(selected_indices)[0]

        #selected_indices = tf.concat([selected_indices,
        #                              tf.zeros(self.cfg[MODE + 'MAX_GT_INSTANCES']-num_valid_nms_boxes, tf.int32)], 0)

        #selected_scores = tf.concat([selected_scores,
        #                              tf.zeros(self.cfg[MODE + 'MAX_GT_INSTANCES']-num_valid_nms_boxes, tf.float32)], -1)

        #boxes = tf.gather(bbox, selected_indices)
    
        #scores = tf.expand_dims(selected_scores, axis=-1)
        #classes = tf.cast(tf.gather(classes, selected_indices), tf.float32)
        #classes = tf.expand_dims(classes, axis=-1) 

        MODE = 'DET_' if self.mode == 'detection' else ''

        boxes, scores, classes, nvalid = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
            scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
            max_output_size_per_class=self.cfg[MODE + 'MAX_GT_INSTANCES'], 
            max_total_size=self.cfg[MODE + 'MAX_GT_INSTANCES'],
            iou_threshold=self.cfg[MODE + 'RPN_IOU_THRESHOLD'],
            score_threshold=self.cfg[MODE + 'RPN_SCORE_THRESHOLD']
        )

        # [y1, x1, y2, x2] --> [x1, y1, x2, y2]

        y1, x1, y2, x2 = tf.split(boxes, 4, axis=-1)
        rpn_roi = tf.concat([x1, y1, x2, y2], axis=-1)

        scores = tf.expand_dims(scores, axis=-1)
        classes = tf.expand_dims(classes, axis=-1)

        return tf.concat([rpn_roi, scores, classes], axis=-1)

    def get_config(self):
        config = super(YoloNMS, self).get_config()
        config.update({'mode': self.mode, 
                       'cfg': self.cfg})        
        return config

# PART E: Build yolo graph ============================================================================

def RpnGraph(input_image, mode, cfg):
    YoloGraph = eval(cfg['BACKBONE']) 
    rpn_fmaps, outs = YoloGraph(input_image, cfg) 
    pd_boxes = YoloBox(cfg, name = 'yolo_box')(outs)
    pd_rpn = YoloNMS(mode, cfg, name = 'yolo_nms')(pd_boxes)    
    return rpn_fmaps, pd_boxes, pd_rpn
  
                
        

        

