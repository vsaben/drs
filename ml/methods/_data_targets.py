# Description: Create YOLO-ready y1
# Function:
#   A. Assign bb2d to anchor id
#   B. Match y1 from to yolo output 

import tensorflow as tf

# DO: Select own anchors (W x H) = 9 (3 x small, medium, large)
# YOLOv3: Kmean => Idea Kmediods on training

# DO: Weight anchors to smaller objects
# DO: Grid cell size
# DO: Adapt max boxes per scale for each scale independently

@tf.function
def transform_targets(y, cfg_mod, size, isdensity=False):
    
    # Input: y = [xmin, ymin, xmax, ymax, dam_cls, occ, tru, cls,             BASE (8)
    #             screenx, screeny, posz, dimx, dimy, dimz, qx, qy, qz, qw]   HEAD (10)

    # A: Create empty y1 target tensors

    anchor_masks = cfg_mod.MASKS
    nscales = tf.gather(tf.shape(anchor_masks), 0)

    grid_small = size // 32                                              # [a] Smallest grid size 
    grid_sizes = grid_small*2**tf.range(nscales)                         # [b] Grid size at each level [Default: 13, 26, 52] (Going backwards)
    
    yshape = tf.shape(y)
    Nbatch = tf.gather(yshape, 0) 
    Ndetect = tf.gather(yshape, 1)
    Nfeatures = tf.gather(yshape, 2)

    if isdensity: Nfeatures = 0
    arr_yt = transform_targets_empty(Nbatch, grid_sizes, anchor_masks, Nfeatures)  

    # Check IF no targets

    Ntar = tf.math.count_nonzero(y[..., 2])

    if tf.greater(Ntar, 0):       
        anchors = cfg_mod.ANCHORS
        y = add_anchor_ids(y, anchors)                                                                           # B: Determine anchor index for each true box 
        arr_yt = transform_targets_for_output(arr_yt, y, grid_sizes, anchor_masks, 
                                              Nbatch, Ndetect, Nfeatures, isdensity)   # C: Populate y target tensors
    
    return divide_tensor_grids(arr_yt, grid_sizes)

def transform_targets_empty(Nbatch, grid_sizes, anchor_masks, Nfeatures):

    # Function: Create scale-amalgamated anchor result zero array
    # Output: (grid_sum, grid_sum, nanchors (at each level), Nfeatures + 1)
    # Notes: Nfeatures = 4 + 1 + 1 + 3 + 10 = 19  
    #                  = bb + obj + dam + features + head

    gridsum = tf.reduce_sum(grid_sizes)
    nmask_pscale = tf.gather(tf.shape(anchor_masks), 1)                 # Assume: Constant    
    return tf.zeros((Nbatch, gridsum, gridsum, nmask_pscale, Nfeatures + 1), tf.float32)

@tf.function
def divide_tensor_grids(arr_yt, grid_sizes):
    
    # Function: Divide result tensor per grid size
    # Output: (size0, size1, ...) 
    # Note: Hard coded for 3 levels

    splits = tf.split(arr_yt, grid_sizes, axis = 1)
    
    for i, split in enumerate(splits):
        splits[i] = tf.split(split, grid_sizes, axis = 2)[i]

    return tuple(splits)

    #if tf.equal(tf.size(grid_sizes), 2):
    #    m, l = tf.split(arr_yt, grid_sizes, axis = 1)
    #
    #    m = tf.split(m, grid_sizes, axis = 2)[0]
    #    l = tf.split(l, grid_sizes, axis = 2)[1]
    #    return tuple([m, l])
  
    #else:
    #    s, m, l = tf.split(arr_yt, grid_sizes, axis = 1)
    #
    #    s = tf.split(s, grid_sizes, axis = 2)[0]
    #    m = tf.split(m, grid_sizes, axis = 2)[1]
    #    l = tf.split(l, grid_sizes, axis = 2)[2]

    #    return tuple([s, m, l])

def add_anchor_ids(y, anchors):   
    anchor_areas = anchors[:, 0] * anchors[:, 1]                         # [b] Anchor pixel areas
    
    box_wh = y[..., 2:4] - y[..., :2]                                    # [c] Box width (xmax - xmin) and height (ymax - ymin) 
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),                         # [d] [nbatch, ndetections, nanchors, 2] 
                        (1, 1, anchors.shape[0], 1))                        
    box_areas = box_wh[..., 0] * box_wh[..., 1]                          # [e] Box areas [ndetections, [bb area] * nanchors]
    
    min_anchor_w = tf.minimum(box_wh[..., 0], anchors[:, 0])             # [f] Minimum (box, anchor) => Find anchor W smaller than standardised box W
    min_anchor_h = tf.minimum(box_wh[..., 1], anchors[:, 1])             #     Minimum (box, anchor) => Find anchor H smaller than standardised box H 
    intersection = min_anchor_w * min_anchor_h                           #     Area of corresponding minimum anchors
                                                                                                         
    iou = intersection / (box_areas + anchor_areas - intersection)       # [g] IOU(bbox, anchor)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)            # [h] ID w/ max IOU
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y = tf.concat([y, anchor_idx], axis=-1)                              # Append best anchor id to y [index = 19]
    return y

def transform_targets_for_output(arr_yt, y, grid_sizes, anchor_masks, Nbatch, Ndetect, Nfeatures, isdensity=False):
    
    # Input : y1 = (N, boxes, (xmin, ymin, xmax, ymax, class, best_anchor))
    # Centered Output: y1_out = (Nbatch, grid, grid, anchors, [x, y, w, h, obj, class])
    # Uncentered Output: y1_out = (Nbatch, grid, grid, anchors, [xmin, ymin, xmax, ymax, obj, class])
    # > Recentered in loss function

    # DO: Avoid overriding boxes

    nlevels = tf.gather(tf.shape(anchor_masks), 0)
    gridsum = tf.constant(0)

    for level in tf.range(nlevels):
        anchor_idxs = tf.gather(anchor_masks, level)
        grid_size = tf.gather(grid_sizes, level)                
    
        for i in tf.range(Nbatch):
            for j in tf.range(Ndetect):
                det = y[i][j]
                if tf.equal(det[2], 0): continue                                      # Check: xmax = 0 (skip padding)              
                anchor_eq = tf.equal(anchor_idxs, tf.cast(det[Nfeatures], tf.int32))  # Check: If bb has an associated anchor scale                                                                              #        at the given prediction level              
                if tf.reduce_any(anchor_eq):                                          # (Refer check above) 
                    box = det[0:4]                                                    # [e] [xmin, ymin, xmax, ymax]
                    box_xy = (det[0:2] + det[2:4]) / 2                                # [f] [xcenter, ycenter]
    
                    anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)               # [g] Anchor scale index
                    grid_xy = tf.cast(box_xy // (1/tf.cast(grid_size, tf.float32)), tf.int32)        # [h] Assign bb to grid cell [13, 26, 52]
                    grid_xy_arr = gridsum + grid_xy
                
                    # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class) --> Assign bb to anchor 

                    if isdensity:
                        arr_yt = update_arr_dt(arr_yt, i, grid_xy_arr, anchor_idx)
                    else:
                        arr_yt = update_arr_yt(arr_yt, i, grid_xy_arr, anchor_idx, Nfeatures, box, det)
                        
        gridsum += grid_size
    return arr_yt    

def update_arr_yt(arr_yt, i, grid_xy_arr, anchor_idx, Nfeatures, box, det):
    index = tf.concat([tf.reshape(tf.tile([i, grid_xy_arr[1], grid_xy_arr[0], anchor_idx[0][0]], [Nfeatures + 1]), (Nfeatures + 1, 4)), 
                       tf.expand_dims(tf.range(Nfeatures + 1), -1)], axis=-1)    
    update = tf.concat([box, tf.constant([1], tf.float32), det[4:-1]], axis = -1)                        
    return tf.tensor_scatter_nd_update(arr_yt, index, update)

def update_arr_dt(arr_dt, i, grid_xy_arr, anchor_idx):
    
    index = [i, grid_xy_arr[1], grid_xy_arr[0], anchor_idx[0][0], 0]
    value = tf.gather(arr_dt, index)
    update = value + 1 
    
    return tf.tensor_scatter_nd_update(arr_dt, index, update) 

# Problem: Overwriting where achor-sharing cars' centroids occur in the same cell
# Concern: Clusters of small vehicles
# Solution: Fine grid sizes (refer _data_grids)