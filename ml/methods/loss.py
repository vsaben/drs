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
import matplotlib.pyplot as plt 

import tensorflow as tf
import tensorflow.keras.losses as KLS
from methods.utils import broadcast_iou

import tensorflow.keras as K 
import tensorflow.keras.layers as KL 
import tensorflow.keras.metrics as KM
import tensorflow.keras.initializers as KI

# PART A: Backbone ======================================================================

def compute_rpn_losses(pd_boxes, gt_boxes, cfg):

    """Calculate RPN loss incl. metric components
    
    :note: accounts for positive and negative ROI through objectness

    :param pd_boxes: 
    :param gt_boxes: [grid_s, grid_m, grid_l] | [grid_m, grid_l]
    :param cfg: base configuration settings

    :result: batch rpn loss
    """

    _anchors = np.array(cfg['ANCHORS'], np.float32)

    losses = []
    for i, pbox in enumerate(pd_boxes):
        anchors = _anchors[cfg['MASKS'][i]]
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

    ignore_mask = tf.cast(best_iou < cfg['RPN_IOU_THRESHOLD'], tf.float32)

    # 6: Calculate losses [nbatch, gridx, gridy, anchors]

    xy_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(gt_xy - pd_xy), axis=-1)
    wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(gt_wh - pd_wh), axis=-1)

    obj_entropy = KLS.binary_crossentropy(y_true=gt_obj, y_pred=pd_obj)
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

class ClsMetricLayer(KL.Layer):

    def __init__(self, cfg, name=None, *args, **kwargs):
        super(ClsMetricLayer, self).__init__(name=name, *args, **kwargs)
        self.cfg = cfg

    def call(self, inputs):

        """Log class losses as metrics, at all abstraction levels"""
        
        pd_boxes, gt_boxes = inputs
        nlevels = len(gt_boxes)
        
        flat_cms = []

        # Per-level

        for lvl in range(nlevels):
            pcls, gcls = self.extract_single_layer_cls(pd_boxes[lvl], gt_boxes[lvl])

            cm = tf.math.confusion_matrix(gcls, pcls, 
                                          num_classes = self.cfg['NUM_CLASSES'], 
                                          name = 'cm_{:d}'.format(lvl))

            flat_cm = tf.cast(tf.reshape(cm, [-1]), tf.float32)
            self.add_cm_metrics(flat_cm, lvl)
            flat_cms.append(flat_cm)

        #all_cls = KL.Lambda(lambda x: compute_all_cls(*x, cfg=self.cfg))([pd_boxes, gt_boxes]) # list([ndetect, [pcls, gcls]])*nlevel

        # Combined

        stack_cm = tf.stack(flat_cms)             # [nlevel, 4]
        flat_cm = tf.reduce_sum(stack_cm, axis=0) # [4]
        self.add_cm_metrics(flat_cm, -1)
     
        return flat_cm

    def extract_single_layer_cls(self, pbox, gbox):

        """Extracts predicted and ground-truth class values
        from yolo grids, at different scales
    
        :param pd_cls: [nbatch, gsize, gsize, plevel, [P(Class_1), ..., P(Class_n)]]
        :param gt_cls: [nbatch, gsize, gsize, plevel, cls integer]

        :result: [nbatch, ndetect, nclass + 1 (class prob + actual cls)]
        """

        _, _, pd_cls, _ = pbox
        _, gt_obj, gt_cls, _ = tf.split(gbox, (4, 1, 1, 10), axis=-1)
    
        obj_mask = tf.squeeze(gt_obj, -1)        # [nbatch, gsize, gsize, plevel]

        pcls_probs = tf.boolean_mask(pd_cls, obj_mask)                # [ndetect, ncls]
        pcls = tf.argmax(pcls_probs, axis=-1)                         # [ndetect]
        gcls = tf.squeeze(tf.boolean_mask(gt_cls, obj_mask), axis=-1) # [ndetect]

        return pcls, gcls

    def add_cm_metrics(self, flat_cm, level):

        suffix = "_{:d}".format(level) if level > -1 else "" 
        
        for i, short_name in enumerate(['tn', 'fn', 'fp', 'tp']):
            
            name = "{:s}".format(short_name) + suffix
            self.add_metric(flat_cm[i], name=name, aggregation='mean')

    def get_config(self):
        config = super(ClsMetricLayer, self).get_config() 
        config.update({'cfg': self.cfg})
        return config
    

    #@classmethod
    #def compute_cm(pcls, gcls, cfg, name=None):

        #dam_mask = tf.squeeze(gcls, axis=-1)
        #cm_weights = (tf.cast(tf.equal(dam_mask, 0), tf.float32)*cfg.DAMAGED_RATIO + 
        #              tf.cast(tf.equal(dam_mask, 1), tf.float32)*(1-cfg.DAMAGED_RATIO))

     #   cm = tf.math.confusion_matrix(gcls, pcls, 
     #                                   num_classes = cfg.NUM_CLASSES, 
        #                              #weights = cm_weights, 
     #                                   name=name)
        #norm_cm = tf.math.around(tf.cast(cm, tf.float32) / tf.reduce_sum(cm, axis=1), 
        #print(norm_cm)
     #   flat_cm = tf.reshape(cm, [-1]) # ADJUST 
     #   return flat_cm

class RpnMetricLayer(KL.Layer):   
    
    def __init__(self, cfg, name=None, *args, **kwargs):
        super(RpnMetricLayer, self).__init__(name=name, *args, **kwargs)
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

        # Bounding Box

        all_rpn_losses = KL.Lambda(lambda x: compute_rpn_losses(*x, self.cfg))([pd_boxes, gt_boxes])

        # Type [t]

        all_tloss = tf.transpose(all_rpn_losses, [0, 2, 1])             # [nbatch, nlosses, nlevels]        
        weighted_tloss = all_tloss * self.lweights                      # [nbatch, nlosses, nlevels]        
        sum_tloss = tf.reduce_sum(weighted_tloss, axis = 2)             # [nbatch, nlosses]        
        mean_tloss = tf.reduce_mean(sum_tloss, axis = 0)                # [nlosses]
        
        ntypes = len(self.tnames)
        for i in range(ntypes):
            self.add_metric(mean_tloss[i], name=self.tnames[i], aggregation="mean") 
        
        # Level [l]

        weighted_lloss = all_rpn_losses * self.tweights                # [nbatch, nlevels, nlosses]
        sum_lloss = tf.reduce_sum(weighted_lloss, axis = 2)            # [nbatch, nlevels]
        mean_lloss = tf.reduce_mean(sum_lloss, axis = 0)               # [nlevels]

        nlevels = len(self.lnames)
        for i in range(nlevels):
            self.add_metric(mean_lloss[i], name=self.lnames[i], aggregation="mean") 

        # Type-Level

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
        config = super(RpnMetricLayer, self).get_config()        
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
     :note: remove padding check depth = 0 (only one possible, index = 7)
     :note: add configuration (cfg) to allow for different loss functions
    """

    #gt_cond = tf.gather(gt_pose, 7, axis = -1)    # [nbatch, 40]
    gt_cond = gt_pose[..., 7]                      # [nbatch, 40]

    positive_ix = tf.where(gt_cond > 0)           # [none, 2] --> (instance index, cond true index)

    pd_pose = tf.gather_nd(pd_pose, positive_ix)  # [none, 10] --> (true instance combined, pose)
    gt_pose = tf.gather_nd(gt_pose, positive_ix)

    pd_center, pd_depth, pd_dims, pd_quart = tf.split(pd_pose, (2, 1, 3, 4), axis = -1)                  # [nbatch, ndetect, 10]  
    gt_rpn, gt_center, gt_depth, gt_dims, gt_quart = tf.split(gt_pose, (5, 2, 1, 3, 4), axis = -1)       # [nbatch, ndetect, 5 + 10]
        
    gt_wh = gt_rpn[..., 2:4] - gt_rpn[..., 0:2]

    center_loss = pose_center_loss(pd_center, gt_center, gt_wh)
    depth_loss = pose_depth_loss(pd_depth, gt_depth) 
    dim_loss = pose_dim_loss(pd_dims, gt_dims)
    quart_loss = pose_quart_loss(pd_quart, gt_quart)

    sublosses = [center_loss, depth_loss, dim_loss, quart_loss]     
    return tf.stack(sublosses)

# Center

def pose_center_loss(pd_center, gt_center, gt_wh):
    
    """
    :note: scale center offset for 2d region size
    :note: imperfect - volume / rotation / visible with surface area 
    :note: consistent with rpn yolo loss

    :param pd_center: [nbatch, num_rois, normalised [x, y]]
    :param gt_center: [nbatch, num_rois, normalised [x, y]]
    """
   
    box_loss_scale = 2 - gt_wh[..., 0] * gt_wh[..., 1]      

    square_loss = tf.square(pd_center - gt_center)                          # [nbatch, ndetect, 2]
    sum_loss = box_loss_scale * tf.reduce_sum(square_loss, axis = -1)       # [nbatch, ndetect]   
    return tf.reduce_mean(sum_loss)                                         # [1]

# Depth

def pose_depth_loss(pd_depth, gt_depth):
    """
    gt_xy, input_image

    :note: l1, l2 - problem: unit depth provides an equal 
    loss contribution btween distant and near points 
    distant should be less than near. propose log of depth errors [l_depth]
    
    :note: logs - problem: depth's step edge structure (prominent in nature)
    logs sensitive to shifts in depth direction but not x, y directions. 
    insensitive to distortion and blur of edges. propose penalise edge errors
    more (gradients of depth) [l_grad]. Wide view (encapsulate vehicle)

    :note: Weight l_depth + lambda * l_grad - problem: l_grad cannot penalise small 
    structural errors (not of concern) 

    ("Depth loss (explain)")

    :note: popular metrics

    :note: Eigen et al. logs with coarse and fine-scale networks. fine-scale refines
    coarse estimation

    :options: mean abs (l1)    | equivalent loss contribution of near and far points  
              mean square (l2) |
              scale-invariant loss (l_eigen) | insensitive to x, y direction
              scale-invariant with gradients (l_eigengrad) [choose initial]
              berHu (l_berhu)  |
              huber (l_huber)  |
              least-squared adversarial
              conditional random fields

    :note: sobel derivative for simplicity
    """

    # adjust to sobel size

    #sobel_x = KI.constant([[1, 0, -1], [2, 0, -2], [1, 0, -1]])    
    #sobel_y = KI.constant([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) 

    #dx = KL.Conv2D(1, 3, kernel_initializer=sobel_x, trainable=False)(input_image)
    #dy = KL.Conv2D(1, 3, kernel_initializer=sobel_y, trainable=False)(input_image)

    #D = tf.math.log(pd_depth) - tf.math.log(gt_depth)      # [nbatch, ndetect, 1]  
    
    D = tf.square(pd_depth - gt_depth)    
    return tf.reduce_mean(D)                               # [1]

# Dimension

def pose_dim_loss(pd_dims, gt_dims):
    # ADJUST: Dimension loss scaling (like bbox)

    square_loss = tf.square(pd_dims - gt_dims)                   # [nbatch, ndetect, 3]
    sum_loss = tf.reduce_sum(square_loss, axis = -1)             # [nbatch, ndetect]
    return tf.reduce_mean(sum_loss)                              # [1]

# Rotation

def pose_quart_loss(pd_quart, gt_quart):
    
    # ADJUST: Add other rotation loss functions
    # 6D-VNet: End-to-end 6DoF Vehicle Pose Estimation from Monocular RGB Images ||q - qhat / ||qhat|| ||1

    sum_loss = tf.norm(gt_quart - pd_quart / tf.norm(pd_quart, axis = -1, keepdims=True), ord = 1, axis = -1) # [nbatch, ndetect]
    return tf.reduce_mean(sum_loss)                                                                           # [1]


class PoseMetricLayer(KL.Layer):
    
    def __init__(self, cfg, name=None, *args, **kwargs): 
        super(PoseMetricLayer, self).__init__(name=name, *args, **kwargs)
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
        config = super(PoseMetricLayer, self).get_config()
        config.update({'cfg': self.cfg})
        return config

# PART C: Combined losses ==========================================================

class CombineLossesLayer(KL.Layer):

    def __init__(self, cfg, name=None, *args, **kwargs):
        super(CombineLossesLayer, self).__init__(name=name, *args, **kwargs)
        self.cfg = cfg
        self.scales = tf.constant(list(cfg['LOSSES'].values()))

    def call(self, inputs):

        rpn_loss, pose_loss, cls_metric = inputs

        losses = tf.stack([rpn_loss, pose_loss])
        final_loss = tf.reduce_sum(losses * self.scales, name = 'loss')
        self.add_loss(final_loss)
        return final_loss

    def get_config(self): 
        config = super(CombineLossesLayer, self).get_config()
        config.update({'cfg': self.cfg})
        return config








 












