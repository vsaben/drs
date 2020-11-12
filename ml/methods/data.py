"""
    Description: Read in, format and batch training and validation data
    Function: Build data pipeline
    - PART A: Read in tfrecord files
    - PART B: Reformat and transform targets
    - PART C: Build pipeline
    - PART D: Summary operations
    -----------------------------------------
    - PART E: Read in single tfrecord
"""

import tensorflow as tf
import math
import os

pi = tf.constant(math.pi)

from methods._data.reformat import (
    get_bb2d, 
    get_screen_center, 
    get_quart, 
    get_vehicle_classes, 
    get_weather_classes, 
    create_cam_rot
)

from methods._data.augment import horizontal_flip

# PART A/B: Read in tfrecord and apply basic reformatting =================

IMAGE_FEATURE_MAP = {
    'image': tf.io.FixedLenFeature([], tf.string),
    
    'pos_x': tf.io.VarLenFeature(tf.float32),
    'pos_y': tf.io.VarLenFeature(tf.float32),
    'pos_z': tf.io.VarLenFeature(tf.float32),
    'dim_x': tf.io.VarLenFeature(tf.float32),
    'dim_y': tf.io.VarLenFeature(tf.float32),
    'dim_z': tf.io.VarLenFeature(tf.float32),
    'rot_x': tf.io.VarLenFeature(tf.float32),
    'rot_y': tf.io.VarLenFeature(tf.float32),
    'rot_z': tf.io.VarLenFeature(tf.float32),

    'tru': tf.io.VarLenFeature(tf.float32),
    'occ': tf.io.VarLenFeature(tf.float32),
    'dam': tf.io.VarLenFeature(tf.int64),
    'cls': tf.io.VarLenFeature(tf.string),

    'cam_rot': tf.io.FixedLenFeature([], tf.string),
    'cam_w': tf.io.FixedLenFeature([], tf.int64),
    'cam_h': tf.io.FixedLenFeature([], tf.int64),
    'cam_vfov': tf.io.FixedLenFeature([], tf.float32),

    'env_time': tf.io.FixedLenFeature([], tf.string), 
    'env_wthr': tf.io.FixedLenFeature([], tf.string),
    'env_rain': tf.io.FixedLenFeature([], tf.int64),
    'env_snow': tf.io.FixedLenFeature([], tf.int64)
    }

def parse_basic_tfrecord(tfrecord, image_size):

    ex = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)

    x = tf.image.decode_jpeg(ex['image'], channels=3)
    x = tf.image.resize(x, (image_size, image_size))
    x /= 255 

    camera = ExtractCamera(ex)
   
    return ex, x, camera

def ExtractCamera(ex):

    cam_rot = tf.io.decode_raw(ex['cam_rot'], tf.float64)
    cam_rot = tf.reshape(cam_rot, (3,))
    fy = 1/tf.tan(ex['cam_vfov']/2) 

    camera = {'rot': cam_rot, 
              'R': create_cam_rot(cam_rot), 
              'w': ex['cam_w'], 
              'h': ex['cam_h'], 
              'fx': tf.cast(ex['cam_h']/ex['cam_w'], tf.float32) * fy, 
              'fy': fy}

    return camera

def parse_detect_tfrecord(tfrecord, image_size):
    ex, x, camera = parse_basic_tfrecord(tfrecord, image_size)
    return x, camera
   
def parse_tfrecord(tfrecord, image_size):
    
    """Parse individual tfrecords and extract image, target, camera and environment features

    :param tfrecord: string tfrecord file path
    :param image_size: resize target image dimensions

    :result: [x, cpos, features, camera, environment] 
    """
    
    ex, x, camera = parse_basic_tfrecord(tfrecord, image_size)

    # Environment property dictionary

    weatherclasses = get_weather_classes(ex['env_wthr'])
    environment =  tf.stack([weatherclasses, 
                              tf.cast(ex['env_rain'], tf.int32), 
                              tf.cast(ex['env_snow'], tf.int32)])
                   #"time": ex['env_time']

    #N = tf.size(ex['pos_x'])    ADJUST

    #if N == 0: 
    #    cpos = tf.zeros((0, 9))
    #    features = tf.zeros((0, 4))
    #    return x, cpos, features, camera, environment

    # Row-wise: [pos_x, pos_y, pos_z, dim_x, dim_y, dim_z, rot_x, rot_y, rot_z]

    cpos = tf.stack([tf.sparse.to_dense(ex['pos_x']), 
                     tf.sparse.to_dense(ex['pos_y']), 
                     tf.sparse.to_dense(ex['pos_z']), 
                     tf.sparse.to_dense(ex['dim_x']), 
                     tf.sparse.to_dense(ex['dim_y']), 
                     tf.sparse.to_dense(ex['dim_z']), 
                     tf.sparse.to_dense(ex['rot_x']), 
                     tf.sparse.to_dense(ex['rot_y']), 
                     tf.sparse.to_dense(ex['rot_z'])], axis=1)

    # Row-wise: [dam, occ, tru, cls]

    vehicleclasses = get_vehicle_classes(ex['cls'])
    features = tf.stack([tf.cast(tf.sparse.to_dense(ex['dam']), tf.float32),
                         tf.sparse.to_dense(ex['occ']),
                         tf.sparse.to_dense(ex['tru']), 
                         tf.sparse.to_dense(vehicleclasses)], axis = 1)

    return x, cpos, features, camera, environment

def transform_prebatch(x, cpos, features, camera, environment, cfg):  

    """Collect and pad y_true
    :note: group additional, non-predicted target features in y_true
    
    :param x: image with standardised pixel values
    :param cpos: [pos_x, pos_y, pos_z, dim_x, dim_y, dim_z, rot_x, rot_y, rot_z] 
    :param features:
    :param camera: camera property dictionary
    :param environment: environment property dictionary 
    :param max_boxes: maximun number of detections per image (refer config)

    :result: [x, y_collect]
    :result y_collect: [y, camera, environment]
    :result y: [xmin, ymin, xmax, ymax, dam status, (BASE)
                occ, tru, veh cls,                  (FEATURES)
                screenx, screeny, posz,             (HEAD)
                dimx, dimy, dimz,                   (HEAD)
                qx, qy, qz, qw] (padded),           (HEAD)
                camera,                             (FEATURES)
                environment                         (FEATURES)
    """

    N = tf.shape(cpos)[0]
    
    #if N > 0:

    bb2d = get_bb2d(camera, cpos)
    features = tf.transpose(features) 
        
    input_rpn = tf.stack([
        
        # Input: RPN [4 + 1 = 5]
        
        bb2d[0],      # xmin
        bb2d[1],      # ymin 
        bb2d[2],      # xmax
        bb2d[3],      # ymax                      
        features[0]], # dam
                         axis = 1)

    screen = get_screen_center(camera, cpos)
    
    dims = cpos[:, 3:6] # [dimx, dimy, dimz]
    if cfg.USE_DIM_ANCHOR:                             
        med_dims = tf.constant(cfg.DIM_ANCHOR, tf.float32)
        dims -= med_dims    
    dims = tf.transpose(dims)

    quart = get_quart(cpos)

    input_pose = tf.stack([

        # Input: Head [2 + 1 + 3 + 4 = 10]

        tf.cast(screen[0], tf.float32), # screenx 
        tf.cast(screen[1], tf.float32), # screeny
        cpos[:, 2],      # posz 
        dims[0],         # dimx
        dims[1],         # dimy
        dims[2],         # dimz
        quart[0],        # ix  
        quart[1],        # iy
        quart[2],        # iz
        quart[3]],       # iw
                          axis = 1)  
        
    #input_features = tf.stack([
    # dam, occ, tru, cls = 0, 1, 2, 3
        # Extra: Features [3]

    #    features[:, occ],
    #    features[:, tru], 
    #    features[:, cls]], axis=1)
        
    #else:
    #    input_rpn = tf.zeros((0, 5))
    #    input_pose = tf.zeros((0, 10))

    paddings = [[0, cfg.MAX_GT_INSTANCES - N], [0, 0]] 

    padded_rpn = tf.pad(input_rpn, paddings)
    padded_pose = tf.pad(input_pose, paddings)

    inputs = {'input_image': x,              
              'input_pose': tf.ensure_shape(padded_pose, (cfg.MAX_GT_INSTANCES, 10)), 
              'input_rpn': tf.ensure_shape(padded_rpn, (cfg.MAX_GT_INSTANCES, 5))} # seems alphabetical
    
    outputs = tf.constant(0) 
              
    return inputs, outputs

# B/C: Further reformat, batch, augment and shuffle the data ========================================

def load_ds(data_path, istrain, cfg):

    """Load, augment and batch training or validation data using tensorflow's dataset iterator
    
    :param data_path: folder containing training and validation tfrecord subfolders
    :param cfg_set: training or validation configuration settings 
    :param cfg_mod: model configuration settings
    :param size: resized, target image size
    :param batch_size: batch size
    :param max_boxes: maximum number of detections per image (refer config)
    :option isaugment: whether a horizontal flip is applied, and the results concatenated

    :result: batched dataset iterator [nbatch, x, [anchor y, camera, environment]] 
    """
    
    ds = read_and_parse(data_path, cfg.IMAGE_SIZE, istrain)                           # Out: [x, cpos, features, camera, environment] 
      
    if istrain and cfg.ISAUGMENT:
        augment_ds = ds.map(horizontal_flip)
        ds = ds.concatenate(augment_ds)

    ds = ds.map(lambda *x: transform_prebatch(*x, cfg))                                # Out: [x, padded input_gt, camera, environment] ADJUST
    
    if istrain: 
        ds = ds.shuffle(buffer_size=cfg.BUFFER_SIZE, reshuffle_each_iteration=True) 

    ds = ds.batch(cfg.BATCH_SIZE)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  
    return ds

def read_and_parse(data_path, image_size, istrain = True):

    """Read in data and perform basic reformatting
    :note: shards - where distributed training (only have 1 gpu) 
    :note: interleave - where perform operations in parallel above parallel_calls 
           (possible training time improvement)
    :note: cache - none as data does not fit into memory 
    :note: repeat - not seem necessary given reshuffle after each iteration

    :param data_path: folder containing training and validation tfrecord subfolders
    :param size: resized, target image size
    :option istrain: training or validation parse

    :result: dataset iterator containing [x, cpos, features, camera, environment]
             per image
    """ 

    ds_set = "train" if istrain else "val"
    regex = os.path.join(data_path, ds_set) + "\\*.tfrecord"
    ds = tf.data.Dataset.list_files(regex)    

    ds = tf.data.TFRecordDataset(ds)
    ds = ds.map(lambda e: parse_tfrecord(e, image_size))    
    return ds

# PART D: Read in a single example ================================================

def load_tfrecord(regex, mode, cfg):

    """Load tfrecord for detection and/or inference

    :param regex: regex distinguishing tfrecord/s
    :param mode: 'inference' (w/ annotations) OR 'detection' (w/o annotations)
    :param cfg: model configuration settings

    :result: [x, camera]      (detection)
             [x, camera, out] (inference)
    :result out: [bbox, score = 1, cls, center, depth, dimensions, quarternions]
    """

    with_anns = (mode in ["inference"])

    ds = tf.data.Dataset.list_files(regex) 
    nimage = tf.data.experimental.cardinality(ds).numpy()

    ds = tf.data.TFRecordDataset(ds)

    if not with_anns:
        ds = ds.map(lambda e: parse_detect_tfrecord(e, cfg.IMAGE_SIZE))
    
    ds = ds.batch(nimage)
    return ds



