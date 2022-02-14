"""
    Description: Loss function control
    Function: 
    - PART A: Backbone (x, y, w, h, dam_cls)
    - PART B: Pose (cx, cy             [center | bbox]
                    posz               [center depth]
                    dimx, dimy, dimz,  [dimensions | anchor]
                    qx, qy, qz, qw)    [rotation]
    - PART C: Combine losses  
"""

import numpy as np

import tensorflow as tf
import tensorflow.keras.losses as KLS
from methods.utils import broadcast_iou

import tensorflow.keras.layers as KL 

# PART A: Backbone ======================================================================

def compute_rpn_losses(pd_boxes, gt_boxes, cfg):

    """Calculate RPN loss incl. metric components
    
    :note: accounts for positive and negative ROI through objectness

    :param pd_boxes: [grid_s, grid_m, grid_l] | [grid_m, grid_l]
    :param gt_boxes: [grid_s, grid_m, grid_l] | [grid_m, grid_l]
    :param cfg: base configuration settings

    :result: batch rpn loss
    """

    _anchors = tf.constant(cfg['ANCHORS'], tf.float32)
    _xyscales =  tf.constant(cfg['XYSCALE'], tf.float32)

    losses = [] 
    for i, pbox in enumerate(pd_boxes):
        anchors = tf.gather(_anchors, cfg['MASKS'][i])       
        loss = single_layer_yolo_loss(pbox, gt_boxes[i], anchors, _xyscales[i], cfg)            
        losses.append(loss) 
    
    return tf.stack(losses, axis=1)                 # [nbatch, nlevels, nlosses]
    
def single_layer_yolo_loss(pbox, gbox, anchors, xyscale_i, cfg):

    # 1: Transform predicted boxes [nbatch, grid, grid, anchors, (x, y, w, h, obj, .. cls)] (grid) 

    pd_box_img, pd_obj, pd_cls, pd_grd = pbox 
    pd_xy = pd_grd[..., 0:2]
    pd_wh = pd_grd[..., 2:4]

    # 2: Convert normalised image gt boxes to grid space [nbatch, grid, grid, anchors, (x1, y1, x2, y2, obj, .. cls, .. features)] (normalised image)

    gt_box_img, gt_obj, gt_cls, gt_pose = tf.split(gbox, (4, 1, 1, 10), axis=-1)   

    gt_box_x1y1 = gt_box_img[..., 0:2]
    gt_box_x2y2 = gt_box_img[..., 2:4]

    gt_img_xy = (gt_box_x1y1 + gt_box_x2y2) / 2
    gt_img_wh = gt_box_x2y2 - gt_box_x1y1

    gt_img = tf.concat([gt_img_xy, gt_img_wh], axis=-1)                # [x1, y1, x2, y2] >> [x, y, w, h]
    
    # > Give a higher weighting to smaller boxes 
    
    box_loss_scale = 2 - gt_img_wh[..., 0] * gt_img_wh[..., 1]
    
    # > Continue: (image) to (grid)
    
    grid_size = tf.shape(gt_img)[1]
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)   

    gt_xy = (gt_img_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32) + 0.5 * (xyscale_i - 1)) / xyscale_i
    gt_wh = tf.math.log(gt_img_wh / anchors)
    gt_wh = tf.where(tf.math.is_inf(gt_wh), tf.zeros_like(gt_wh), gt_wh)  

    # 5: Calculate masks

    obj_mask = tf.squeeze(gt_obj, axis=-1)

    best_iou = tf.map_fn(lambda x: 
                            tf.reduce_max(                                                      # Ignore FP when iou > threshold
                                broadcast_iou(
                                    x[0], 
                                    tf.boolean_mask(x[1], tf.cast(x[2], tf.bool))), axis=-1),   # Prediction grids matching ground truth
                            (pd_box_img, gt_box_img, obj_mask), tf.float32) 

    ignore_mask = tf.cast(best_iou < cfg['RPN_IOU_THRESHOLD'], tf.float32, name = 'ignore_mask')

    # 6: Calculate losses [nbatch, gridx, gridy, anchors] (based on relative grid coordinates)

    xy_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(gt_xy - pd_xy), axis=-1)
    wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(gt_wh - pd_wh), axis=-1)

    obj_entropy = KLS.binary_crossentropy(gt_obj, pd_obj)
    objcf_loss = obj_mask * obj_entropy
    nobjcf_loss = (1 - obj_mask) * ignore_mask * obj_entropy       
    obj_loss = objcf_loss + nobjcf_loss        
        
    cls_mask_unweighted = tf.squeeze(gt_cls, -1) # [nbatch, gsize, gsize, plevel_anchors] / make more universal
    
    cls_mask_undam = tf.cast(tf.equal(cls_mask_unweighted, 0), tf.float32)*cfg['DAMAGED_RATIO']**cfg['FOCAL_LOSS_ALPHA']
    cls_mask_dam = tf.cast(tf.equal(cls_mask_unweighted, 1), tf.float32)*(1 - cfg['DAMAGED_RATIO'])**cfg['FOCAL_LOSS_ALPHA']
    cls_mask = cls_mask_undam + cls_mask_dam     # [nbatch, gsize, gsize, plevel_anchors]

    class_loss = obj_mask * cls_mask * KLS.sparse_categorical_crossentropy(y_true=gt_cls, y_pred=pd_cls)   
    
    sublosses = [xy_loss, wh_loss, obj_loss, class_loss, objcf_loss, nobjcf_loss]

    # 6. Sum losses over grid [nbatch, 1]

    reduced_losses = []
    for lss in sublosses:
        subloss = tf.reduce_sum(lss, axis = (1, 2, 3)) 
        reduced_losses.append(subloss)

    return tf.stack(reduced_losses, axis=1) # [nbatch, 1, 6]
    
class RpnLossLayer(KL.Layer):   
    
    def __init__(self, cfg, name=None):
        super(RpnLossLayer, self).__init__(name=name)
        self.cfg = cfg

        nlevels = len(cfg['MASKS'])
        
        lnames, lweights = list(zip(*cfg['RPN_LVL_LOSSES'].items()))
        self.lnames = lnames[:nlevels]                       # level names
        self.lweights = tf.constant(lweights[:nlevels])      # level weights across types 
        
        tnames, tweights = list(zip(*cfg['RPN_TYPE_LOSSES'].items()))
        self.tnames = tnames                                 # type names
        self.tweights = tf.constant(tweights)                # type weights across levels

    def call(self, inputs):
        """ Log all computed rpn losses as metrics, at all levels of abstraction incl.
        type, level and type-level.
        
        :note: 'aggregation' = how to aggregate the per-batch values over each epoch
        
        :return: batch rpn loss
        """
        
        pd_boxes, gt_boxes = inputs

        # Losses

        all_rpn_losses = KL.Lambda(lambda x: compute_rpn_losses(*x, self.cfg))([pd_boxes, gt_boxes])

        # > Type [t]

        all_tloss = tf.transpose(all_rpn_losses, [0, 2, 1])             # [nbatch, nlosses, nlevels]        
        weighted_tloss = all_tloss * self.lweights                      # [nbatch, nlosses, nlevels]        
        sum_tloss = tf.reduce_sum(weighted_tloss, axis = 2)             # [nbatch, nlosses]                                  
        mean_tloss = tf.reduce_mean(sum_tloss, axis = 0)                # [nlosses]
        
        ntypes = len(self.tnames)
        for i in range(ntypes):
            self.add_metric(mean_tloss[i], name=self.tnames[i], aggregation="mean") 
        
        # > Level [l]

        weighted_lloss = all_rpn_losses * self.tweights                # [nbatch, nlevels, nlosses]
        sum_lloss = tf.reduce_sum(weighted_lloss, axis = 2)            # [nbatch, nlevels]
        mean_lloss = tf.reduce_mean(sum_lloss, axis = 0)               # [nlevels]

        nlevels = len(self.lnames)
        for i in range(nlevels):
            self.add_metric(mean_lloss[i], name=self.lnames[i], aggregation="mean") 

        # > Type-Level

        for lvl in range(nlevels):
            for typ in range(ntypes): 
                mean_tlloss = tf.reduce_mean(all_rpn_losses[:, lvl, typ])                
                lname = self.extract_subloss_shortname(self.lnames[lvl])
                tname = self.extract_subloss_shortname(self.tnames[typ])                
                tlname = 'rpn_{:s}_{:s}_loss'.format(lname, tname) 
                self.add_metric(mean_tlloss, name=tlname, aggregation="mean")

        # Final
                
        rpn_loss = tf.reduce_sum(mean_tloss * self.tweights, name='rpn_loss')
        return rpn_loss

    def get_config(self):    
        config = super(RpnLossLayer, self).get_config()        
        config.update({'cfg': self.cfg})        
        return config

    @staticmethod
    def extract_subloss_shortname(name):

        _1 = name.find('_') + 1
        _2 = name.rfind('_')

        return name[_1:_2]

# PART B: Pose ========================================================================

def compute_pose_losses(pd_pose, gt_pose, cfg):
    """
     :note: only positive ROIs contribute to loss (similar to bbox in mask-rcnn)
     :note: remove padding check y2 = 0 (only one possible)
     :note: add configuration (cfg) to allow for different loss functions
    """

    gt_cond = gt_pose[..., 2]  
    positive_ix = tf.where(gt_cond > 0)            # [none, 2] --> (instance index, cond true index)
    
    pd_pose = tf.gather_nd(pd_pose, positive_ix)   # [none, 10] --> (true instance combined, pose) 
    gt_pose = tf.gather_nd(gt_pose, positive_ix)

    def compute_pose_loss(pd_pose, gt_pose, cfg):

        pd_center, pd_depth, pd_dims, pd_quart = tf.split(pd_pose, (2, 1, 3, 4), axis = -1)                  # [nbatch, ndetect, 10]  
        gt_rpn, gt_center, gt_depth, gt_dims, gt_quart = tf.split(gt_pose, (4, 2, 1, 3, 4), axis = -1)       # [nbatch, ndetect, 4 + 10]

        center_loss = pose_center_loss(pd_center, gt_center, gt_rpn)
        depth_loss = pose_depth_loss(pd_depth, gt_depth)
        dim_loss = pose_dim_loss(pd_dims, gt_dims)
        quart_loss = pose_quart_loss(pd_quart, gt_quart)

        return tf.stack([center_loss, depth_loss, dim_loss, quart_loss])

    loss = tf.cond(tf.size(gt_pose) == 0,
                   lambda: tf.zeros(4, tf.float32),
                   lambda: compute_pose_loss(pd_pose, gt_pose, cfg))

    return loss

def pose_center_loss(pd_center, gt_center, gt_rpn):
    
    """Recollect relative, predicted center coordinates in normalised 
    image coordinates. Compute MSE.

    :note: scale center offset for 2d region size (box_loss_scale) 
    :note: consistent with rpn yolo loss

    :param pd_center: [nbatch, num_rois, normalised [x, y]]
    :param gt_center: [nbatch, num_rois, normalised [x, y]]
    """

    gt_wh = gt_rpn[..., 2:4] - gt_rpn[..., 0:2]
    box_loss_scale = 2 - gt_wh[..., 0] * gt_wh[..., 1]  

    square_loss = tf.square(pd_center - gt_center)                          # [nbatch, ndetect, 2]
    sum_loss = box_loss_scale * tf.reduce_sum(square_loss, axis = -1)       # [nbatch, ndetect]   

    return tf.reduce_mean(sum_loss)                                         # [1]

def pose_depth_loss(pd_depth, gt_depth):
    """Depth loss. Squared logarithmic difference.

    :note: penalises nearby losses more

    :param pd_depth: predicted depth
    :param gt_depth: ground-truth depth  

    """

    log_pd_depth = tf.math.log(pd_depth)
    log_gt_depth = tf.math.log(gt_depth)

    D = tf.square(log_pd_depth - log_gt_depth)                           # [nbatch, ndetect, 1] 
    return tf.reduce_mean(D)                                             # [1]

def pose_dim_loss(pd_dims, gt_dims):
    
    """Dimension loss. Calculates MSE of absolute or 
    relative dimensions (if USE_DIM_ANCHOR = T).
    
    :note: scale for 3D box volume
    :note: MSE of relative difference = MSE of absolute difference
    :note: gt encompasses median dimension specification

    :param pd_dims:
    :param gt_dims:
    """
    
    square_loss = tf.square(pd_dims - gt_dims)                   # [nbatch, ndetect, 3]
    sum_loss = tf.reduce_sum(square_loss, axis = -1)             # [nbatch, ndetect]
    return tf.reduce_mean(sum_loss)                              # [1]

# Rotation

def pose_quart_loss(pd_quart, gt_quart):
    
    # ADJUST: Add other rotation loss functions
    # 6D-VNet: End-to-end 6DoF Vehicle Pose Estimation from Monocular RGB Images ||q - qhat / ||qhat|| ||1

    sum_loss = tf.norm(gt_quart - pd_quart / tf.norm(pd_quart, axis = -1, keepdims=True), ord = 1, axis = -1) # [nbatch, ndetect]
    return tf.reduce_mean(sum_loss)                                                                           # [1]

class PoseLossLayer(KL.Layer):
    
    def __init__(self, cfg, name=None, *args, **kwargs): 
        super(PoseLossLayer, self).__init__(name=name, *args, **kwargs)
        self.cfg = cfg
        
        pnames, pweights = list(zip(*cfg['POSE_SUBLOSSES'].items())) 
        self.pnames = pnames
        self.pweights = tf.constant(pweights)

    def call(self, inputs):
        """Log all computed pose losses as metrics. 
        Includes: center, depth, dimension and rotation
        
        :param pd_head:
        :param gt_head:

        """
        pd_pose, gt_pose = inputs
        all_pose_losses = KL.Lambda(lambda x: compute_pose_losses(*x, self.cfg))([pd_pose, gt_pose]) # [nbatch, nlosses]

        # Sublosses

        nlosses = len(self.pnames)
        for i in range(nlosses):
            self.add_metric(all_pose_losses[i], name=self.pnames[i], aggregation='mean')

        # Final

        pose_loss = tf.reduce_sum(all_pose_losses * self.pweights, name='pose_loss')
        return pose_loss

    def get_config(self): 
        config = super(PoseLossLayer, self).get_config()
        config.update({'cfg': self.cfg})
        return config

# PART C: Combined losses ==========================================================

class CombineLossesLayer(KL.Layer):

    def __init__(self, cfg, verbose, name=None, *args, **kwargs):
        super(CombineLossesLayer, self).__init__(name=name, *args, **kwargs)
        self.cfg = cfg
        self.verbose = verbose
        self.scales = tf.constant(list(cfg['LOSSES'].values()))

    def call(self, inputs):
        rpn_loss, pose_loss = inputs[:2] if self.verbose else inputs
        losses = tf.stack([rpn_loss, pose_loss])
        final_loss = tf.reduce_sum(losses * self.scales, name = 'loss')
        self.add_loss(final_loss)
        return final_loss
        
    def get_config(self): 
        config = super(CombineLossesLayer, self).get_config()
        config.update({'cfg': self.cfg, 
                       'verbose': self.verbose})
        return config








 












