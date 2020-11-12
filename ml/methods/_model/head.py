"""
    Description: Model head architectures and supporting functions
    Functions:
    - PART A: ROI Alignment
    - PART B: Supporting functions
    - PART C: Head configurations
    - PART D: Build DRS head
"""

import numpy as np

import tensorflow as tf
import tensorflow.keras as K 
import tensorflow.keras.layers as KL    
   
from methods._data.targets import add_anchor_ids, divide_grids 

# PART A: ROI Align ================================================================ 
 
def assign_roi_level(anchor_id, nlevels):
    """Assigns each roi to a feature map level based on its anchor-mask
    assignment

    :note: replaces mask-rcnn's log2 roi feature map assignment

    :param anchor_id: [0, ..., 8] (3 levels) | [0, ..., 5] (2 levels)
    :param nlevels: number of yolo prediction layers

    :result: [nbatch, ndetect, 1]
             values [0, 1, 2] | [0, 1]
    """

    return tf.math.ceil(nlevels - anchor_id / nlevels) - 1

def pyramid_roi_align(rois, fmaps, cfg):
    """Implements ROIAlign Pooling on multiple levels of the yolo feature pyramid
    
    :note: allow gradient propagation through proposals if cfg.USE_RPN_ROIS

    :param rois: [nbatch, cfg.MAX_GT_INSTANCES, (x1, y1, x2, y2)] in normalized coordinates.
    :param fmaps: List of pyramid feature maps, each is [batch, height, width, channels]. 
                  Starts at [nbatch, 13, 13, 256] by default.
    :param pool_shape: [pool_height, pool_width] of the output pooled regions. Default [7, 7] 
     
    :result: pooled regions [nbatch, cfg.MAX_GT_INSTANCES, pool_height, pool_width, channels]
    """
        
    anchors = np.array(cfg['ANCHORS'], np.float32)
    masks = cfg['MASKS']
    nlevels = len(masks)
   
    # Assign each ROI to a level in the pyramid based on the ROI area. 
    # Account for: normalised bbox co-ordinates  
    # Assume: Square image size
         
    anchor_id = add_anchor_ids(rois, anchors, ids_only = True) # [nbatch, maxboxes, 1]

    x1, y1, x2, y2 = tf.split(rois, 4, axis=2)
    boxes = tf.concat([y1, x1, y2, x2], axis=-1) # [nbatch, maxboxes, 4] 

    roi_level = assign_roi_level(anchor_id, nlevels) # [nbatch, maxboxes, 1]

    # Loop through levels and apply ROI pooling to each. P0 to P2.
        
    pool_shape = [cfg['POOL_SIZE'], cfg['POOL_SIZE']]

    pooled = []
    box_to_level = []

    for level in range(nlevels):
        
        eq_ix = tf.squeeze(tf.equal(roi_level, level), axis=-1) # [nbatch, cfg.MAX_GT_INSTANCES]
        ix = tf.where(eq_ix)                                    # [ndetect_across_nbatch, batch_id + eq_index]
        
        level_boxes = tf.gather_nd(boxes, ix)                   # [ndetect_across_nbatch, 4]

        # Box indices for crop_and_resize.
        box_indices = tf.cast(ix[:, 0], tf.int32)               # [ndetect_across_nbatch]

        # Keep track of which box is mapped to which level
        box_to_level.append(ix)

        # Stop gradient propogation to ROI proposals [ ADJUST ]
        #level_boxes = tf.stop_gradient(level_boxes) 
        #box_indices = tf.stop_gradient(box_indices) 

        # Crop and Resize
        # From Mask R-CNN paper: "We sample four regular locations, so
        # that we can evaluate either max or average pooling. In fact,
        # interpolating only a single value at each bin center (without
        # pooling) is nearly as effective."
        #
        # Here we use the simplified approach of a single value per bin,
        # which is how it's done in tf.crop_and_resize()
        # Result: [batch * num_boxes, pool_height, pool_width, channels]           

        roi_aligned = tf.image.crop_and_resize(fmaps[level], 
                                               level_boxes, 
                                               box_indices, 
                                               pool_shape,
                                               method="bilinear")
        pooled.append(roi_aligned)

    # Pack pooled features into one tensor
    pooled = tf.concat(pooled, axis=0)

    # Pack box_to_level mapping into one array and add another
    # column representing the order of pooled boxes
    box_to_level = tf.concat(box_to_level, axis=0)
    box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
    box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],axis=1)

    # Rearrange pooled features to match the order of the original boxes
    # Sort box_to_level by batch then box index
    # TF doesn't have a way to sort by two columns, so merge them and sort.
    sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
    ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0]).indices[::-1]
    ix = tf.gather(box_to_level[:, 2], ix)
    pooled = tf.gather(pooled, ix)

    # Re-add the batch dimension
    shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)    
    pooled = tf.reshape(pooled, shape)
    return pooled

# PART B: Supporting functions =================================================================

def TDConv(x, filters, kernel, strides = 1, padding = 'valid'):
    x = KL.TimeDistributed(KL.Conv2D(filters, kernel, strides = strides, padding = padding))(x)
    x = KL.TimeDistributed(KL.BatchNormalization())(x)
    x = KL.LeakyReLU()(x)    
    return x


# PART C: Head module configurations ============================================================

def PoseGraph(cfg, name=None):
    
    """Builds head pose graph

    :param roi_align: regions of interest (refer model input)  

    :result: [nbatch, ndetect, 10]
    """

    x = input = KL.Input([cfg['MAX_GT_INSTANCES'], cfg['POOL_SIZE'], cfg['POOL_SIZE'], cfg['TOP_DOWN_PYRAMID_SIZE']])

    x = TDConv(x, cfg['FC_LAYERS_SIZE'], 7)
    x = TDConv(x, cfg['FC_LAYERS_SIZE'], 1) 
    shared = tf.squeeze(x, (2, 3))          # [nbatch, cfg.MAX_GT_INSTANCES, FC_LAYERS_SIZE]

    out = KL.TimeDistributed(KL.Dense(10), name='pose_out')(shared)  

    return K.Model([input], out, name=name)

# PART D: Build DRS head ========================================================================

def RoiAlign(cfg, name=None):
    
    """Computes regions of interest

    :param rois: image area of interest
    :param feature_maps: output level feature maps

    :result: [nbatch, ndetect, 7, 7, TOP_DOWN_PYRAMID_SIZE]
    """
    
    rois = KL.Input([cfg['MAX_GT_INSTANCES'], 4])

    nscale = len(cfg['MASKS'])
    grid_small = cfg['IMAGE_SIZE'] // 32             # smallest grid size 

    grid_sizes = [grid_small*2**i for i in range(nscale)]
    out_sizes = [int(128*2**(nscale - 1)*0.5**i) for i in range(nscale)]

    feature_maps = [KL.Input([gsize, gsize, osize]) for gsize, osize in zip(grid_sizes, out_sizes)] 

    filters = cfg['TOP_DOWN_PYRAMID_SIZE']

    fmaps = []    
    for i, P in enumerate(feature_maps):
        fpn_name = 'fpn_p{:d}'.format(i)
        fmaps.append(KL.Conv2D(filters, (3, 3), padding = 'same', name = fpn_name)(P))

    roi_align = pyramid_roi_align(rois, fmaps, cfg)

    return K.Model([rois, feature_maps], roi_align, name=name)

class DetectionTargetsLayer(KL.Layer):
    """Generates bbox, class and pose annotations for rpn generated proposals
    on a per image basis. Pads the result.
    """

    def __init__(self, cfg, name=None):
        super(DetectionTargetsLayer, self).__init__(name=name)
        self.cfg = cfg

    def call(self, inputs):
        """
        :param rpn_roi: [nbatch, cfg.MAX_GT_INSTANCES, 4]
        :param gt_boxes: [grid_s, grid_m, grid_l] | [grid_m, grid_l]
        """
                
        rpn_roi, gt_boxes = inputs
        rpn_roi = tf.stop_gradient(rpn_roi)

        masks = tf.constant(self.cfg.MASKS, tf.int32)
        nlevels = tf.shape(masks)[0]

        grid_small = self.cfg.IMAGE_SIZE // 32                                              
        grid_sizes = grid_small*2**tf.range(nlevels)                         
    
        nbatch = tf.shape(rpn_rois)[0]
        ndetect = self.cfg.MAX_GT_INSTANCES  

        anchors = tf.constant(self.cfg.ANCHORS, tf.float32)
        rpn_roi = add_anchor_ids(rpn_roi, anchors)

        gt_pose_od = tf.zeros((nbatch, ndetect, 15))

        for i in tf.range(nbatch):
            jsum = 0
            for j in tf.range(ndetect):
                det = rpn_roi[i][j]
                if tf.equal(det[2], 0): continue
                det_anchor_id = det[-1]
                
                for level in tf.range(nlevels):
                    anchor_idxs = masks[level]
                    grid_size = grid_sizes[level]                                                                           
                    anchor_eq = tf.equal(anchor_idxs, tf.cast(det_anchor_id, tf.int32))
                    
                    if tf.reduce_any(anchor_eq):                                          
                        box = det[0:4]                                                    
                        box_xy = (det[0:2] + det[2:4]) / 2                               
    
                        anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)               
                        grid_xy = tf.cast(box_xy // (1/tf.cast(grid_size, tf.float32)), tf.int32)        

                        roi_features = gt_boxes[level][i, grid_xy[1], grid_xy[0], anchor_idx[0][0]][4:]
                        update = tf.concat([box, roi_features], axis=-1)
                        gt_pose_od = tf.tensor_scatter_nd_update(gt_pose_od, [i, jsum], update)
                        jsum += 1

        gt_od, gt_pose = tf.split(gt_pose_od, (5, 10))       
        return gt_pose, gt_od

class OutLayer(KL.Layer):

    def __init__(self, name=None):
        super(OutLayer, self).__init__(name=name)

    def call(self, inputs):

        """
        :param nms: [boxes, scores, classes, nvalid]


        :param pd_pose: [nbatch, cfg.MAX_GT_INSTANCES, 10]


        :result: [nbatch, ndetect, boxes (4) + score (1) + class (1) + [RPN]
                       center (2) + depth (1) + dims (3) + quart (4) [HEAD]]
        """

        nms, pd_pose = inputs
        
        rpn_roi, scores, classes, nvalid = nms

        scores = tf.expand_dims(scores, axis=-1)
        classes = tf.expand_dims(classes, axis=-1)

        padded_out = tf.concat([rpn_roi, scores, classes, pd_pose], axis=-1, name='out_padded') # [nbatch, cfg.MAX_GT_INSTANCES, 16]
        mask = tf.not_equal(tf.gather(padded_out, 2, axis=-1), 0, name='out_mask')              # [nbatch, cfg.MAX_GT_INSTANCES]      
        out = tf.ragged.boolean_mask(padded_out, mask, name='out')               
        return nvalid, out

#def build_drs_head(rois, feature_maps, cfg):
    """Builds the computation graph of the mask head of Feature Pyramid Network.
    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    """

    # Process feature maps

 #   filters = cfg.TOP_DOWN_PYRAMID_SIZE

 #   fmaps = []
    
#    for i, P in enumerate(feature_maps):
#        fpn_name = 'fpn_p{:d}'.format(i)
#        fmaps.append(Conv2D(filters, (3, 3), padding = 'same', name = fpn_name)(P))

    # ROI align

#    x = pyramid_roi_align(rois, fmaps, cfg) # [nbatch, ndetect, 7, 7, TOP_DOWN_PYRAMID_SIZE]
#    print(x)

    # Modules
    
#    pose = pose_graph(x, cfg)                # [nbatch, ndetect, nfeatures]    
#    print(pose) 
    
#    if cfg.ADD_HEAD_OD == "separate": 
#        od = od_graph(x, cfg)           # Add to output [ADJUST]       
#        print(od) 

#    return pose # Model([rois, feature_maps], out, name=name)
  
 
      


    
