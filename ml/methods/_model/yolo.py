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

import numpy as np

import tensorflow as tf
import tensorflow.keras as K 
import tensorflow.keras.layers as KL 

# PART A: Individual layers and blocks ================================================

def DarknetConv(x, filters, kernel, strides=1, batch_norm=True, activate_type = 'leaky'):
    if strides == 1:
        padding = 'same'
    else:
        x = KL.ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = KL.Conv2D(filters=filters, kernel_size=kernel,
                  strides=strides, padding=padding,
                  use_bias=not batch_norm)(x)             # , kernel_regularizer=l2(0.0005) [ADJUST]
    if batch_norm:
        x = KL.BatchNormalization()(x)
        if activate_type == 'leaky': x = KL.LeakyReLU(alpha=0.1)(x)
        if activate_type == 'mish': x = mish(x)
    return x

def mish(x):
    return KL.Lambda(lambda x: x * tf.math.tanh(tf.math.softplus(x)))(x)

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

def Darknet(cfg, name=None):
    input_image = KL.Input((cfg['IMAGE_SIZE'], cfg['IMAGE_SIZE'], cfg['CHANNELS']))
    x = DarknetConv(input_image, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  
    x = x_36 = DarknetBlock(x, 256, 8)  
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return K.Model(input_image, (x_36, x_61, x), name=name)

def CSPDarknet(cfg, name=None):
    input_image = KL.Input((cfg['IMAGE_SIZE'], cfg['IMAGE_SIZE'], cfg['CHANNELS']))
    x = DarknetConv(input_image, 32, 3, activate_type = 'mish')
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
    return K.Model(input_image, (x_1, x_2, x), name=name)

def DarknetTiny(cfg, name=None):
    input_image = KL.Input((cfg['IMAGE_SIZE'], cfg['IMAGE_SIZE'], cfg['CHANNELS']))
    x = DarknetConv(input_image, 16, 3)
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
    return K.Model(input_image, (x_8, x), name=name)

def CSPDarknetTiny(cfg, name=None):
    input_image = KL.Input((cfg['IMAGE_SIZE'], cfg['IMAGE_SIZE'], cfg['CHANNELS']))
    x = DarknetConv(input_image, 32, 3, 2)
    x = DarknetConv(x, 64, 3, 2)
    x = DarknetConv(x, 64, 3)
    x = CSPDarknetBlockTiny(x, 32)
    x = CSPDarknetBlockTiny(x, 64)
    x, rt1 = CSPDarknetBlockTiny(x, 128, return_rt_last = True)
    return K.Model(input_image, (rt1, x), name=name)

# PART C: Yolo head supporting functions ==================================================

def YoloConv(filters, iscsp=False, name=None): 
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            x, x_skip = inputs = KL.Input(x_in[0].shape[1:]), KL.Input(x_in[1].shape[1:])                        
            x = DarknetConv(x, filters, 1)
            if iscsp: x_skip = DarknetConv(x_skip, filters, 1)
            x = KL.UpSampling2D(2)(x)
            x = KL.Concatenate()([x, x_skip])
        else:
            x = inputs = KL.Input(x_in.shape[1:])                         

        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        return K.Model(inputs, x, name=name)(x_in)
    return yolo_conv

def CSPYoloConv(filters, name=None):
    def csp_yolo_conv(x_in):    
        x, rt = inputs = KL.Input(x_in[0].shape[1:]), KL.Input(x_in[1].shape[1:])
        x = DarknetConv(x, filters, 3, 2)
        x = KL.Concatenate()([x, rt])
        x = YoloConv(x, filters)
        return K.Model(inputs, x, name=None)(x_in)
    return csp_yolo_conv

def YoloConvTiny(filters, name=None): 
    def yolo_conv_tiny(x_in):
        if isinstance(x_in, tuple):
            x, x_skip = inputs = KL.Input(x_in[0].shape[1:]), KL.Input(x_in[1].shape[1:])
            x = DarknetConv(x, filters, 1)
            x = KL.UpSampling2D(2)(x)
            x = KL.Concatenate()([x, x_skip])
        else:
            x = inputs = KL.Input(x_in.shape[1:])
            x = DarknetConv(x, filters, 1)
        return K.Model(inputs, x, name=name)(x_in)
    return yolo_conv_tiny

def YoloOutput(filters, anchors, classes = 2, name=None):    
    def yolo_output(x_in):
        x = inputs = KL.Input(x_in.shape[1:])
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
        x = KL.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)))(x)
        return K.Model(inputs, x, name=name)(x_in)
    return yolo_output

def Boxes(pred, anchors, classes = 2):

    """Extract bounding boxes, objectness scores and class probabilities from 
    yolo predictions
    :note: adapt undamaged/damaged class output into a probability of damage
    
    :param pred: (nbatch, grid, grid, (x, y, w, h, P(damage)))
    :param anchors: scale-specific bounding box anchors
    :param classes: number of class predictions (refer note)

    :result bbox_xyxy: [nbatch, grid, grid, 3, 4] (normalised)
    :result objectness: [nbatch, grid, grid, 3, 1]
    :result class_probs: [nbatch, grid, grid, 3, nclass]
    :result pred_box_xywh: [nbatch, grid, grid, 3, 4] (original)
    """
    
    box_xy, box_wh, objectness, class_probs = tf.split(pred, (2, 2, 1, classes), axis=-1) 

    # Final layer output transformations

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box_xywh = tf.concat((box_xy, box_wh), axis=-1)                      

    # Recollect normalised bbox (x1, y1), (x2, y2) co-ordinates 
   
    grid_size = tf.shape(pred)[1:3]
    grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))   # grid[x][y] == (y, x)
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)               # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox_xyxy = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox_xyxy, objectness, class_probs, pred_box_xywh

# PART D: Yolo =========================================================================

def YoloV3(cfg):

    masks = cfg['MASKS']
    classes = cfg['NUM_CLASSES']
    
    input_image = KL.Input((cfg['IMAGE_SIZE'], cfg['IMAGE_SIZE'], cfg['CHANNELS']))
    x_36, x_61, x = Darknet(cfg, name='darknet')(input_image)

    x = P0 = YoloConv(512, name='yolo_conv_0')(x)
    out_0  = YoloOutput(512, len(masks[0]), classes, name='yolo_out_0')(x)

    x = P1 = YoloConv(256, name='yolo_conv_1')((x, x_61))
    out_1  = YoloOutput(256, len(masks[1]), classes, name='yolo_out_1')(x)

    x = P2 = YoloConv(128, name='yolo_conv_2')((x, x_36))
    out_2  = YoloOutput(128, len(masks[2]), classes, name='yolo_out_2')(x)

    fmaps  = (P0, P1, P2)
    outs   = (out_0, out_1, out_2) 

    return K.Model(input_image, (fmaps, outs), name='yolo')       
    
def YoloV3T(cfg):

    masks = cfg['MASKS']
    classes = cfg['NUM_CLASSES']
    
    input_image = KL.Input((cfg['IMAGE_SIZE'], cfg['IMAGE_SIZE'], cfg['CHANNELS']))
    x_8, x = DarknetTiny(name='darknet')(input_image)

    x = P0 = YoloConvTiny(256, name='yolo_conv_0')(x)
    out_0  = YoloOutput(256, len(masks[0]), classes, name='yolo_out_0')(x)

    x = P1 = YoloConvTiny(128, name='yolo_conv_1')((x, x_8))
    out_1  = YoloOutput(128, len(masks[1]), classes, name='yolo_out_1')(x)

    fmaps = (P0, P1)
    outs  = (out_0, out_1)

    return K.Model(input_image, (fmaps, outs), name = 'yolo')

def YoloV4(cfg):
    
    masks = cfg['MASKS']
    classes = cfg['NUM_CLASSES']
    
    input_image = KL.Input((cfg['IMAGE_SIZE'], cfg['IMAGE_SIZE'], cfg['CHANNELS']))
    rt1, rt2, rt0 = CSPDarknet(name='yolo_darknet')(input_image)

    x = rt0
    x = rt2 = YoloConv(256, name='yolo_conv_00', iscsp = True)((x, rt2))

    x = P2 = YoloConv(128, name='yolo_conv_0', iscsp = True)((x, rt1))    
    out_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_out_2')(x)
    
    x = P1 = CSPYoloConv(256, name='yolo_conv_1')((x, rt2))
    out_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_out_1')(x)

    x = P0 = CSPYoloConv(512, name = 'yolo_conv_0')((x, rt0))
    out_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_out_0')(x)

    fmaps = (P0, P1, P2)
    outs = (out_0, out_1, out_2)

    return K.Model(input_image, (fmaps, outs), name='yolo')

def YoloV4T(cfg):

    masks = cfg['MASKS']
    classes = cfg['NUM_CLASSES']
    
    input_image = KL.Input((cfg['IMAGE_SIZE'], cfg['IMAGE_SIZE'], cfg['CHANNELS']))            
    rt1, x = CSPDarknetTiny(name='darknet')(input_image)

    x = P0 = YoloConvTiny(256, name='yolo_conv_0')(x)
    out_0 = YoloOutput(256, len(masks[0]), classes, name='yolo_out_0')(x)
  
    x = P1 = YoloConvTiny(128, name='yolo_conv_1')((x, rt1))
    out_1 = YoloOutput(128, len(masks[1]), classes, name='yolo_out_1')(x)

    fmaps = (P0, P1)
    outs = (out_0, out_1)       

    return K.Model(input_image, (fmaps, outs), name='yolo')

class YoloBox(KL.Layer):

    def __init__(self, cfg, name=None, *args, **kwargs):
        super(YoloBox, self).__init__(name=name, *args, **kwargs)
        self.cfg = cfg
        
        self.anchors = np.array(cfg['ANCHORS'], np.float32)
        self.nlevels = len(cfg['MASKS'])

    def call(self, outs):             
        boxes = [Boxes(outs[i], self.anchors[self.cfg['MASKS'][i]], self.cfg['NUM_CLASSES']) 
                 for i in range(self.nlevels)]
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
    
        nms_boxes = [box[:3] for box in inputs]

        # Combine and reshape bbox (bbox), objectness / confidence (conf) and class probablities / type (clsp)
        # ADJUST: Concat / rpn_roi

        bbox = tf.concat([tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])) for o in nms_boxes], axis=1)
        conf = tf.concat([tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])) for o in nms_boxes], axis=1)
        clsp = tf.concat([tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])) for o in nms_boxes], axis=1)

        scores = conf * clsp

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

        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        rpn_roi = tf.concat([x1, y1, x2, y2], axis=-1)

        return rpn_roi, scores, classes, nvalid

    def get_config(self):
        config = super(YoloNMS, self).get_config()
        config.update({'mode': self.mode, 
                       'cfg': self.cfg})        
        return config

# PART E: Build yolo graph ============================================================================

def YoloGraph(input_image, mode, cfg):
    yolo_graph = eval(cfg['BACKBONE']) 
    rpn_fmaps, outs = yolo_graph(cfg)(input_image)
    pd_boxes = YoloBox(cfg, name = 'yolo_box')(outs)
    nms = YoloNMS(mode, cfg, name = 'yolo_nms')(pd_boxes)    
    return rpn_fmaps, pd_boxes, nms
  
                
        

        

