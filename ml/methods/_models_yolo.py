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
    input_image = KL.Input((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.CHANNELS))
    x = DarknetConv(input_image, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  
    x = x_36 = DarknetBlock(x, 256, 8)  
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return K.Model(input_image, (x_36, x_61, x), name=name)

def CSPDarknet(cfg, name=None):
    input_image = KL.Input((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.CHANNELS))
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
    input_image = KL.Input((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.CHANNELS))
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
    input_image = KL.Input((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.CHANNELS))
    x = DarknetConv(input_image, 32, 3, 2)
    x = DarknetConv(x, 64, 3, 2)
    x = DarknetConv(x, 64, 3)
    x = CSPDarknetBlockTiny(x, 32)
    x = CSPDarknetBlockTiny(x, 64)
    x, rt1 = CSPDarknetBlockTiny(x, 128, return_rt_last = True)
    return K.Model(input_image, (rt1, x), name=name)

# PART C: Yolo head supporting functions ==================================================

def YoloConv(input, filters, iscsp=False, name=None):               
    if isinstance(input, tuple):
        x, x_skip = input                          
        x = DarknetConv(x, filters, 1)
        if iscsp: x_skip = DarknetConv(x_skip, filters, 1)
        x = KL.UpSampling2D(2)(x)
        x = KL.Concatenate()([x, x_skip])
    else:
        x = input                          

    x = DarknetConv(x, filters, 1)
    x = DarknetConv(x, filters * 2, 3)
    x = DarknetConv(x, filters, 1)
    x = DarknetConv(x, filters * 2, 3)
    x = DarknetConv(x, filters, 1)
    return x                               

def CSPYoloConv(input, filters):    
    x, rt = input
    x = DarknetConv(x, filters, 3, 2)
    x = KL.Concatenate()([x, rt])
    x = YoloConv(x, filters)
    return x

def YoloConvTiny(input, filters):        
    if isinstance(input, tuple):
        x, x_skip = input
        x = DarknetConv(x, filters, 1)
        x = KL.UpSampling2D(2)(x)
        x = KL.Concatenate()([x, x_skip])
    else:
        x = input
        x = DarknetConv(x, filters, 1)
    return x

def YoloOutput(input, filters, anchors, classes = 2, name=None):    
    x = DarknetConv(input, filters * 2, 3)
    x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
    x = KL.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)))(x)
    return x

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


def NMS(outputs, anchors, masks, classes, max_boxes, iou_thresh, score_thresh):
    
    # Combine and reshape bbox (bbox), objectness / confidence (conf) and class probablities / type (clsp)

    bbox = tf.concat([tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])) for o in outputs], axis=1)
    conf = tf.concat([tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])) for o in outputs], axis=1)
    clsp = tf.concat([tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])) for o in outputs], axis=1)

    scores = conf * clsp

    boxes, scores, classes, nvalid = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=max_boxes, 
        max_total_size=max_boxes,
        iou_threshold=iou_thresh,
        score_threshold=score_thresh
    )

    return boxes, scores, classes, nvalid

def extract_anchors_masks(cfg_mod):
    anchors = cfg_mod.ANCHORS.numpy()
    masks = cfg_mod.MASKS.numpy().tolist()    
    return anchors, masks

# PART D: Yolo =========================================================================

def YoloV3(cfg):

    _, masks = extract_anchors_masks(cfg.YOLO)
    classes = cfg.NUM_CLASSES
    
    input_image = KL.Input((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.CHANNELS))
    x_36, x_61, x = Darknet(cfg, name='darknet')(input_image)

    x = P0 = YoloConv(x, 512, name='yolo_conv_0')
    out_0  = YoloOutput(x, 512, len(masks[0]), classes, name='yolo_out_0')

    x = P1 = YoloConv((x, x_61), 256, name='yolo_conv_1')
    out_1  = YoloOutput(x, 256, len(masks[1]), classes, name='yolo_out_1')

    x = P2 = YoloConv((x, x_36), 128, name='yolo_conv_2')
    out_2  = YoloOutput(x, 128, len(masks[2]), classes, name='yolo_out_2')

    fmaps  = (P0, P1, P2)
    outs   = (out_0, out_1, out_2) 

    return K.Model(input_image, (fmaps, outs), name='yolo')       
    
def YoloV3T(cfg):

    _, masks = extract_anchors_masks(cfg.YOLO)
    classes = cfg.NUM_CLASSES
    
    input_image = KL.Input((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.CHANNELS))
    x_8, x = DarknetTiny(name='darknet')(input_image)

    x = P0 = YoloConvTiny(256, name='yolo_conv_0')(x)
    out_0  = YoloOutput(256, len(masks[0]), classes, name='yolo_out_0')(x)

    x = P1 = YoloConvTiny(128, name='yolo_conv_1')((x, x_8))
    out_1  = YoloOutput(128, len(masks[1]), classes, name='yolo_out_1')(x)

    fmaps = (P0, P1)
    outs  = (out_0, out_1)

    return K.Model(input_image, (fmaps, outs), name = 'yolo')

def YoloV4(cfg):
    
    _, masks = extract_anchors_masks(cfg.YOLO)
    classes = cfg.NUM_CLASSES
    
    input_image = KL.Input((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.CHANNELS))
    rt1, rt2, rt0 = CSPDarknet(name='yolo_darknet')(input_image)

    x = rt0
    x = rt2 = YoloConv(256, name='yolo_conv_00', iscsp = True)((x, rt2))

    x = P2 = YoloConv(128, name='yolo_conv_0', iscsp = True)((x, rt1))    
    out_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_out_2')(x)
    
    x = P1 = CSPYoloConv(256, name='yolo_conv_1')([x, rt2])
    out_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_out_1')(x)

    x = P0 = CSPYoloConv(512, name = 'yolo_conv_0')([x, rt0])
    out_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_out_0')(x)

    fmaps = (P0, P1, P2)
    outs = (out_0, out_1, out_2)

    return K.Model(input_image, (fmaps, outs), name='yolo')

def YoloV4T(cfg):

    _, masks = extract_anchors_masks(cfg.YOLO)
    classes = cfg.NUM_CLASSES
    
    input_image = KL.Input((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.CHANNELS))            
    rt1, x = CSPDarknetTiny(name='darknet')(input_image)

    x = P0 = YoloConvTiny(256, name='yolo_conv_0')(x)
    out_0 = YoloOutput(256, len(masks[0]), classes, name='yolo_out_0')(x)
  
    x = P1 = YoloConvTiny(128, name='yolo_conv_1')((x, rt1))
    out_1 = YoloOutput(128, len(masks[1]), classes, name='yolo_out_1')(x)

    fmaps = (P0, P1)
    outs = (out_0, out_1)       

    return K.Model(input_image, (fmaps, outs), name='yolo')

def YoloBoxes(outs, cfg, name = None):

    anchors, masks = extract_anchors_masks(cfg.YOLO)    
    nlevels = len(masks)

    boxes = [Boxes(outs[i], anchors[masks[i]], cfg.NUM_CLASSES) for i in range(nlevels)]
    return boxes
    

def YoloNMS(outs, cfg):
       
    anchors, masks = extract_anchors_masks(cfg.YOLO)    
    # ADJUST: training/inference

    boxes = YoloBoxes(outs, cfg) 
    nms_boxes = [box[:3] for box in boxes]
    nms = NMS(nms_boxes, anchors, masks, cfg.NUM_CLASSES, 
                                         cfg.MAX_GT_INSTANCES, 
                                         cfg.RPN_IOU_THRESHOLD, 
                                         cfg.RPN_SCORE_THRESHOLD)
    return boxes, nms


# PART E: Build yolo graph ============================================================================

def build_yolo_graph(input_image, cfg):
    _, masks = extract_anchors_masks(cfg.YOLO)
    yolo_graph = eval(cfg.YOLO.NAME) 
    rpn_fmaps, outs = yolo_graph(cfg)(input_image)     
    pd_boxes, nms = KL.Lambda(lambda x: YoloNMS(x, cfg), name='yolo_nms')(outs)
    return rpn_fmaps, pd_boxes, nms

    # ADJUST
            #if cfg.USE_RPN_ROI:            # Ignore predicted ROIs and use ROIs provided as an input / #          
        #    nms_rois = nms[0]            
        #    rpn_rois, gt_features = extract_roi_pd(nms_rois, input_gt, cfg)
        #    nvalid = nms[3] 
        #else:

        

