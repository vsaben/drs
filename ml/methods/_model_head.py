"""
    Description: Model head architectures and supporting functions
    Functions:
    - PART A: ROI Alignment
    - PART B: Supporting functions
    - PART C: Head configurations
    - PART D: Build DRS head
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Input, 
    Lambda, 
    LeakyReLU,
    TimeDistributed
)

from methods._models_yolo import extract_anchors_masks
from methods._data_targets import add_anchor_ids, divide_grids 

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
        
    anchors, masks = extract_anchors_masks(cfg.YOLO)
    nlevels = len(masks)
   
    # Assign each ROI to a level in the pyramid based on the ROI area. 
    # Account for: normalised bbox co-ordinates  
    # Assume: Square image size
         
    anchor_id = add_anchor_ids(rois, anchors, ids_only = True) # [nbatch, maxboxes, 1]

    x1, y1, x2, y2 = tf.split(rois, 4, axis=2)
    boxes = tf.concat([y1, x1, y2, x2], axis=-1) # [nbatch, maxboxes, 4] 

    roi_level = assign_roi_level(anchor_id, nlevels) # [nbatch, maxboxes, 1]

    # Loop through levels and apply ROI pooling to each. P0 to P2.
        
    pool_shape = [cfg.POOL_SIZE, cfg.POOL_SIZE]

    pooled = []
    box_to_level = []

    for level in range(nlevels):
        
        eq_ix = tf.squeeze(tf.equal(roi_level, level))         # [nbatch, cfg.MAX_GT_INSTANCES]
        ix = tf.where(eq_ix)                                   # [ndetect_across_nbatch, batch_id + eq_index]
        
        level_boxes = tf.gather_nd(boxes, ix)                  # [ndetect_across_nbatch, 4]

        # Box indices for crop_and_resize.
        box_indices = tf.cast(ix[:, 0], tf.int32)              # [ndetect_across_nbatch]

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

def TDConv(x, filters, kernel, strides = 1, padding = 'valid', name = None, number = 0):
    conv_name, bn_name = ["{:s}_{:s}{:d}".format(name, t, number) for t in ["conv", "bn"]]
    x = TimeDistributed(Conv2D(filters, kernel, strides = strides, padding = padding), name=conv_name)(x)
    x = TimeDistributed(BatchNormalization(), name=bn_name)(x)
    x = LeakyReLU()(x)    
    return x


# PART C: Head module configurations ============================================================

def single_module(x, cfg):
    
    """
    :result: [nbatch, ndetect, 1, 10]
    """
    
    x = TDConv(x, cfg.FC_LAYERS_SIZE, 7, name = "head_pose", number = 1)
    x = TDConv(x, cfg.FC_LAYERS_SIZE, 1, name = "head_pose", number = 2) 
    shared = tf.squeeze(x, (2, 3)) 
    return shared

# PART D: Output layers =========================================================================

def drs_pose(shared, cfg):
    """
    :result: [nbatch, ndetect, 10]
    """
    x = TimeDistributed(Dense(10), name='head_pose_out')(shared)
    return x 


# PART E: Build DRS head ========================================================================

def build_drs_head(rois, feature_maps, cfg):
    """Builds the computation graph of the mask head of Feature Pyramid Network.
    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    """

    # Process feature maps

    filters = cfg.TOP_DOWN_PYRAMID_SIZE

    fmaps = []
    
    for i, P in enumerate(feature_maps):
        fpn_name = 'fpn_p{:d}'.format(i)
        fmaps.append(Conv2D(filters, (3, 3), padding = 'same', name = fpn_name)(P))

    # ROI align

    x = pyramid_roi_align(rois, fmaps, cfg) # [nbatch, ndetect, 7, 7, TOP_DOWN_PYRAMID_SIZE]
    print(x)

    # Modules
    
    x = single_module(x, cfg)          # [nbatch, ndetect, FC_LAYERS_SIZE] 
    print(x) 
    
    out = drs_pose(x, cfg)             # [nbatch, ndetect, nfeatures]    
    print(out) 

    return out # Model([rois, feature_maps], out, name=name)
  
 
      


    

