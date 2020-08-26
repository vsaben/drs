# Description: Loss function control
# Function: Loss
#   A. 2D BB (x, y, w, h)
#   B. Position
#   C. Dimensions 
#   D. Rotation 

import tensorflow as tf

from methods._models_yolo_head import Boxes
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)
from .utils import broadcast_iou

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

def Loss(anchors, setting = 'total', ignore_thresh = 0.5):
    ind = LOSS_TABLE[setting]
    return YoloComponentLoss(anchors, ind, ignore_thresh)

LOSS_TABLE = {'total': 0,
              'xy': 1,
              'wh': 2, 
              'obj': 3,
              'obj_pos': 4, 
              'obj_neg': 5, 
              'class': 6}

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

# metrics = [xy_loss, wh_loss, obj_loss, obj_loss_conf, noobj_loss_conf, class_loss]
