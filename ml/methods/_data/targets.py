"""
    Description: Transform bounding box annotations into yolo's grid format 
    Function:
    - PART A: Extract grids from bbox annotations [bbox --> grids]
    - PART B: Recover bbox annotations from grids [grids --> bbox]  
"""

import tensorflow as tf

_false = tf.constant(False, tf.bool)

# PART A: BBOX >>> GRIDS =======================================================

def transform_targets(input_rpn, input_pose, cfg):
        
     """Creates scale-specific grids from bbox annotations, with respect to 
     configuration anchors
  
     :param input_rpn: [nbatch, cfg.MAX_GT_INSTANCES, [x1, y1, x2, y2, dam_cls]]
     :param input_pose: [nbatch, cfg.MAX_GT_INSTANCES, [screenx, screeny, posz, 
                                              dimx, dimy, dimz, qx, qy, qz, qw]]
     :param cfg: configuration settings    
      
     :result: (grid_s, grid_m, grid_l) | (grid_m, grid_l)
     :result grid: [nbatch, gsize, gsize, nscale_plevel, 
                                   [bbox (4), obj (1), cls_prob (1), pose (10)]] 
     """

     # Create empty grids

     y = tf.concat([input_rpn, input_pose], axis=-1, name='combine_input')              

     masks = tf.constant(cfg['MASKS'], tf.int32)     
     nlevel = tf.shape(masks)[0]
     nscale = tf.shape(masks)[-1]

     grid_small = cfg['IMAGE_SIZE'] // 32             # smallest grid size 
     grid_sizes = grid_small*2**tf.range(nlevel)                          

     nbatch = tf.shape(y)[0] 
     nfeatures = tf.shape(y)[-1]
     arr_yt = create_empty_grids(nbatch, grid_sizes, nscale, nfeatures)  

     # Update grids with target features (if targets exist)

     ntargets = tf.math.count_nonzero(y[..., 2]) 
     
     def update_arr(cfg, y, arr_yt, grid_sizes):
        anchors = tf.constant(cfg['ANCHORS'], tf.float32)
        y = add_anchor_ids(y, anchors)                                         
        arr_yt = transform_targets_for_output(arr_yt, y, grid_sizes, cfg)
        return arr_yt

     arr_yt = tf.cond(ntargets > 0, 
                      lambda: update_arr(cfg, y, arr_yt, grid_sizes), 
                      lambda: arr_yt)
     
     return divide_grids(arr_yt, grid_sizes)

def create_empty_grids(nbatch, grid_sizes, nscale, nfeatures):

    """Create zero, result anchor-grid array
    :note: assume nscale is constant

    :param nbatch: number of images per batch
    :param grid_sizes: grid size at each prediction level [default: 13, 26, 52]
    :param nscale: number of mask scales at each prediction level (refer note)
    :param nfeatures: rpn [5] + pose [10] = 15 
    
    :result: amalgamated empty grids across grid size axes (1, 2) 
             last dim = nfeatures + 1 to allow for objectness
    """

    gridsum = tf.reduce_sum(grid_sizes)
    return tf.zeros((nbatch, gridsum, gridsum, nscale, nfeatures + 1), 
                    tf.float32, 
                    name = 'empty_grid_array')  

def divide_grids(arr_yt, grid_sizes):
    
    """Divide amalgamated grid into separate grids per prediction level
    
    :param arr_yt: [nbatch, gridsum, gridsum, nscale, nfeatures] (refer create empty grids)
    :param grid_sizes: grid size at each prediction level [default: 13, 26, 52]
    
    :result: (grid_s, grid_m ,grid_l) | (grid_m, grid_l)
    """

    splits = tf.split(arr_yt, grid_sizes, axis = 1)

    for i, split in enumerate(splits):
        splits[i] = tf.split(split, grid_sizes, axis = 2)[i]

    return tuple(splits)

def add_anchor_ids(y, anchors, ids_only = _false):   

    """Assign each bbox to its closest anchor

    :param y: [nbatch, cfg.MAX_GT_INSTANCES, rpn [5] + pose [10]]
    :param anchors: calculated or default model anchors
    :option ids_only: whether to concatenate anchor ids to y  

    :result: [nbatch, cfg.MAX_GT_INSTANCES, rpn [5] + pose [10] + ids [1]]
             [nbatch, cfg.MAX_GT_INSTANCES, ids [1]] if ids_only 
    """
    # B: Determine anchor index for each true box

    anchor_areas = anchors[..., 0] * anchors[..., 1]                     # [b] Anchor pixel areas
    
    box_wh = y[..., 2:4] - y[..., 0:2]                                   # [c] Box width (xmax - xmin) and height (ymax - ymin) 
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),                         # [d] [nbatch, ndetections, nanchors, 2] 
                        (1, 1, tf.shape(anchors)[0], 1))                        
    box_areas = box_wh[..., 0] * box_wh[..., 1]                          # [e] Box areas [ndetections, [bb area] * nanchors]
    
    min_anchor_w = tf.minimum(box_wh[..., 0], anchors[..., 0])           # [f] Minimum (box, anchor) => Find anchor W smaller than standardised box W
    min_anchor_h = tf.minimum(box_wh[..., 1], anchors[..., 1])           #     Minimum (box, anchor) => Find anchor H smaller than standardised box H 
    intersection = min_anchor_w * min_anchor_h                           #     Area of corresponding minimum anchors
                                                                                                         
    iou = intersection / (box_areas + anchor_areas - intersection)       # [g] IOU(bbox, anchor)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)            # [h] ID w/ max IOU

    result = tf.cond(ids_only, 
                   lambda: anchor_idx, 
                   lambda: tf.concat([y, tf.expand_dims(anchor_idx, axis=-1)], axis=-1))          # Append best anchor id to y [index = 20]   
    return result            

@tf.function
def transform_targets_for_output(arr_yt, y, grid_sizes, cfg):
    
    # C: Populate y target tensors
    # Input : y1 = (N, boxes, (xmin, ymin, xmax, ymax, class, best_anchor))
    # Uncentered Output: y1_out = (Nbatch, grid, grid, anchors, [xmin, ymin, xmax, ymax, obj, class])
    # > Recentered in loss function

    # CHECKED: Avoid overriding boxes

    masks = tf.constant(cfg['MASKS'], tf.int32)
    nlevels = tf.shape(masks)[0]

    nbatch = tf.shape(y)[0]
    ndetect = cfg['MAX_GT_INSTANCES']

    gridsum = tf.constant(0)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size = True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size = True)
    idx = 0

    for level in tf.range(nlevels):
        anchor_idxs = masks[level]
        grid_size = grid_sizes[level]
        
        for i in tf.range(nbatch):
            for j in tf.range(ndetect):
                det = y[i][j]                
                if det[2] > 0:                                                            # Check: xmax = 0 (skip padding)                                                                                               
                    anchor_eq = tf.equal(anchor_idxs, tf.cast(det[-1], tf.int32))         # Check: If bb has an associated anchor scale                                                                              #        at the given prediction level                             
                    if tf.reduce_any(anchor_eq):                                          # (Refer check above) 
                        box = det[:4]                                                     # [e] [xmin, ymin, xmax, ymax]
                        box_xy = (det[:2] + det[2:4]) / 2                                 # [f] [xcenter, ycenter]

                        anchor_idx = tf.cast(tf.where(anchor_eq)[0][0], tf.int32)               # [g] Anchor scale index

                        grid_xy = tf.cast(box_xy // (1/tf.cast(grid_size, tf.float32)), tf.int32)        # [h] Assign bb to grid cell [13, 26, 52]
                        grid_xy_arr = gridsum + grid_xy                    

                        # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class) --> Assign bb to anchor 
                    
                        #index = [i, grid_xy_arr[1], grid_xy_arr[0], anchor_idx]                        
                        #update = tf.concat([box, tf.constant([1], tf.float32), det[4:-1]], axis = -1)                        
                        #arr_yt = update_arr_yt(arr_yt, index, update)

                        indexes = indexes.write(idx, [i, grid_xy_arr[1], grid_xy_arr[0], anchor_idx])
                        update = tf.squeeze(tf.split(box, 4) + [tf.constant([1], tf.float32)] + tf.split(det[4:-1], 11))
                        updates = updates.write(idx, update)
                        idx += 1
                        
        gridsum += grid_size
    #return arr_yt
    return tf.tensor_scatter_nd_update(arr_yt, indexes.stack(), updates.stack())    

def update_arr_yt(arr_yt, index, update):
    
    nfeatures = tf.size(update)
    nindex = len(index)

    index_axis = tf.reshape(tf.tile(index, [nfeatures]), [nfeatures, nindex])
    tensor_range = tf.expand_dims(tf.range(nfeatures), -1)
    indices = tf.concat([index_axis, tensor_range], axis=-1)

    return tf.tensor_scatter_nd_update(arr_yt, indices, update)


# Problem: Overwriting where achor-sharing cars' centroids occur in the same cell
# Concern: Clusters of small vehicles
# Solution: Fine grid sizes (refer _data_grids)



    

  

