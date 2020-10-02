"""
    Description: Loss function control
    Function: 
    - PART A: Backbone (x, y, w, h, dam_cls)
    - PART B: Pose (cx, cy             [center | bbox]
                    posz               [center depth]
                    dimx, dimy, dimz,  [dimensions | anchor]
                    qx, qy, qz, qw)    [rotation]
    - PART C: Object detection (x, y, w, h, P(Damage))    
"""

import tensorflow as tf
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)
from methods.utils import broadcast_iou
from methods._models_yolo import extract_anchors_masks

import tensorflow.keras as K 
import tensorflow.keras.layers as KL 

# PART A: Backbone ======================================================================

def compute_rpn_losses(pd_boxes, gt_boxes, cfg):

    """Calculate RPN loss incl. metric components
    
    :param pd_boxes: 
    :param gt_boxes: [grid_s, grid_m, grid_l] | [grid_m, grid_l]
    :param cfg: base configuration settings

    :result: batch rpn loss
    """

    _anchors, masks = extract_anchors_masks(cfg.YOLO)

    losses = []
    for i, pbox in enumerate(pd_boxes):
        anchors = _anchors[masks[i]]
        loss = single_layer_yolo_loss(pbox, gt_boxes[i], anchors, i, cfg)       
        losses.append(loss) 

    return tf.stack(losses, axis=1)                 # [nbatch, nlevels, nlosses]
    
def single_layer_yolo_loss(pbox, gbox, anchors, level, cfg):

    # 1: Transform predicted boxes [nbatch, grid, grid, anchors, (x, y, w, h, obj, .. cls)] 

    pd_box, pd_obj, pd_cls, pd_xywh = pbox
    pd_xy = pd_xywh[..., 0:2]
    pd_wh = pd_xywh[..., 2:4]

    # 2: Transform ground truth boxes [nbatch, grid, grid, anchors, (x1, y1, x2, y2, obj, .. cls, .. features)]

    gt_box, gt_obj, gt_cls, gt_pose = tf.split(gbox, (4, 1, 1, 10), axis=-1)   

    gt_xy = (gt_box[..., 0:2] + gt_box[..., 2:4]) / 2
    gt_wh = gt_box[..., 2:4] - gt_box[..., 0:2]

    # 3: Give a higher weighting to smaller boxes [ADJUST]
        
    box_loss_scale = 2 - gt_wh[..., 0] * gt_wh[..., 1]

    # 4: Recollect ground truth bbox in image coordinates 
    
    grid_size = tf.shape(gbox)[1]
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        
    gt_xy = gt_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)
    gt_wh = tf.math.log(gt_wh / anchors)
    gt_wh = tf.where(tf.math.is_inf(gt_wh), tf.zeros_like(gt_wh), gt_wh)

    # 5: Calculate masks

    obj_mask = tf.squeeze(gt_obj, -1)

    best_iou = tf.map_fn(lambda x: 
                            tf.reduce_max(                                                      # Ignore FP when iou > threshold
                                broadcast_iou(
                                    x[0], 
                                    tf.boolean_mask(x[1], tf.cast(x[2], tf.bool))), axis=-1),
                            (pd_box, gt_box, obj_mask), tf.float32) 

    ignore_mask = tf.cast(best_iou < cfg.RPN_IOU_THRESHOLD, tf.float32)

    # 6: Calculate losses [nbatch, gridx, gridy, anchors]

    xy_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(gt_xy - pd_xy), axis=-1)
    wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(gt_wh - pd_wh), axis=-1)

    obj_entropy = binary_crossentropy(y_true=gt_obj, y_pred=pd_obj)
    objcf_loss = obj_mask * obj_entropy
    nobjcf_loss = (1 - obj_mask) * ignore_mask * obj_entropy       
    obj_loss = objcf_loss + nobjcf_loss        
        
    class_loss = obj_mask * sparse_categorical_crossentropy(y_true=gt_cls, y_pred=pd_cls)   
    
    sublosses = [xy_loss, wh_loss, obj_loss, class_loss, objcf_loss, nobjcf_loss]

    # 6. Sum losses over grid [nbatch, 1]

    reduced_losses = []
    for lss in sublosses:
        subloss = tf.reduce_sum(lss, axis = (1, 2, 3))
        reduced_losses.append(subloss)
     
    return tf.stack(reduced_losses, axis=1) # [nbatch, 1, 6]

# PART B: Pose ========================================================================

def compute_pose_losses(pd_pose, gt_pose, cfg):

    pd_center, pd_depth, pd_dims, pd_quart = tf.split(pd_pose, (2, 1, 3, 4), axis = -1)                  # [nbatch, ndetect, 10]  
    gt_center, gt_depth, gt_dims, gt_quart = tf.split(gt_pose, (2, 1, 3, 4), axis = -1)                  # [nbatch, ndetect, 10]

    center_loss = pose_center_loss(pd_center, gt_center)
    depth_loss = pose_depth_loss(pd_depth, gt_depth, cfg) # cfg added to allow for different functions to be used
    dim_loss = pose_dim_loss(pd_dims, gt_dims)
    quart_loss = pose_quart_loss(pd_quart, gt_quart, cfg)

    sublosses = [center_loss, depth_loss, dim_loss, quart_loss]     
    return tf.stack(sublosses)

# Center

def pose_center_loss(pd_center, gt_center):
    # ADJUST: gt scaling    
    square_loss = tf.square(pd_center - gt_center)               # [nbatch, ndetect, 2]
    sum_loss = tf.reduce_sum(square_loss, axis = -1)             # [nbatch, ndetect]   
    return tf.reduce_mean(sum_loss)                              # [1]

# Depth

def pose_depth_loss(pd_depth, gt_depth, cfg):
    # ADJUST: least square adversial (depth loss)
    square_loss = tf.square(pd_depth - gt_depth)                 # [nbatch, ndetect, 1]
    return tf.reduce_mean(square_loss)                           # [1]

# Dimension

def pose_dim_loss(pd_dims, gt_dims):
    # ADJUST: Dimension loss scaling (like bbox)
    # ADJUST: View x, y, z dim losses

    square_loss = tf.square(pd_dims - gt_dims)                   # [nbatch, ndetect, 3]
    sum_loss = tf.reduce_sum(square_loss, axis = -1)             # [nbatch, ndetect]
    return tf.reduce_mean(sum_loss)                              # [1]

# Rotation

def pose_quart_loss(pd_quart, gt_quart, cfg):
    
    # ADJUST: Add other rotation loss functions
    # 6D-VNet: End-to-end 6DoF Vehicle Pose Estimation from Monocular RGB Images ||q - qhat / ||qhat|| ||1

    sum_loss = tf.norm(gt_quart - pd_quart / tf.norm(pd_quart, axis = -1, keepdims=True), ord = 1, axis = -1) # [nbatch, ndetect]
    return tf.reduce_mean(sum_loss)                                                                           # [1]
 
# PART C: Object Detection ========================================================================

#if cfg.ADD_POSE_OD:
    #    bbox_loss = pose_bbox_loss(pd_bbox, gt_bbox)
    #    class_loss = pose_class_loss(pd_class, gt_class)
    #    sublosses += [bbox_loss, class_loss]

















"""
def YoloComponentLoss(anchors, ind = 0, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):

        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))

        pred_box, pred_obj, pred_prob_dam, pred_xywh = Boxes(y_pred, anchors)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))

        true_box, true_obj, true_class, _ = tf.split(y_true, (4, 1, 1, 13), axis=-1)  # ADJUST   

        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1] # INVESTIGATE

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks

        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask),
            tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)

        obj_entropy = binary_crossentropy(true_obj, pred_obj)
        obj_loss_conf = obj_mask * obj_entropy
        noobj_loss_conf = (1 - obj_mask) * ignore_mask * obj_entropy       
        obj_loss = obj_loss_conf + noobj_loss_conf        
        
        class_loss = obj_mask * tf.reduce_sum(tf.square(true_class - pred_prob_dam), axis = -1)  # Change loss ito 2 classes [Do weight damage class higher] 

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        obj_loss_conf = tf.reduce_sum(obj_loss_conf, axis=(1, 2, 3))
        noobj_loss_conf = tf.reduce_sum(noobj_loss_conf, axis=(1, 2, 3))

        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))
        loss = xy_loss + wh_loss + obj_loss + class_loss 

        losses = [loss, xy_loss, wh_loss, obj_loss, obj_loss_conf, noobj_loss_conf, class_loss]
        
        return losses[ind]
    return yolo_loss



def loss_dict(cfg_mod, ignore_thresh, isloss=True):
    
    res_dict = {}
    masks = cfg_mod.MASKS.numpy().tolist()
    anchors = cfg_mod.ANCHORS.numpy()

    for i, mask in enumerate(masks):
        key = 'yolo_output_{:d}'.format(i)
        if isloss:
            val = Loss(anchors[mask], 'total', ignore_thresh)
        else:
            val = val_metric(anchors, mask, ignore_thresh)
        res_dict[key] = val

    return res_dict

def val_metric(anchors, mask, ignore_thresh):

    res_list = []
    metric_keys = [key for key in LOSS_TABLE.keys() if key != 'total']

    for key in metric_keys:
        fn = Loss(anchors[mask], key, ignore_thresh)
        fn.__name__ = key + "_loss"
        res_list.append(fn)

    return res_list
"""
