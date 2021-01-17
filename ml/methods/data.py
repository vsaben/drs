"""
    Description: Read in, format and batch training and validation data
    Function: Build data pipeline
    - PART A: Read in tfrecord files (and apply basic reformatting)
    - PART B: Reformat and transform targets
    - PART C: Build pipeline
    - PART D: Summary operations
    -----------------------------------------
    - SECTION 1: Train
    - SECTION 2: Detect
    - SECTION 3: Inference
"""

import tensorflow as tf
import math
import os

pi = tf.constant(math.pi)

from methods._data.reformat import (
    get_bb2d, 
    get_rel_yaw,
    get_screen_center, 
    get_vehicle_classes, 
    get_weather_classes, 
    create_cam_rot, 
    E2Q
)

from methods._data.augment import horizontal_flip

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

DETECT_FEATURE_MAP = {
    'image': tf.io.FixedLenFeature([], tf.string),

    'cam_rot': tf.io.FixedLenFeature([], tf.string),
    'cam_w': tf.io.FixedLenFeature([], tf.int64),
    'cam_h': tf.io.FixedLenFeature([], tf.int64),
    'cam_vfov': tf.io.FixedLenFeature([], tf.float32)
    }

def decode_and_resize_jpeg(ex_image, image_size):

    """Decode, resize and normalise tfrecord image"""

    x = tf.image.decode_jpeg(ex_image, channels=3)
    x = tf.image.resize(x, (image_size, image_size))
    x /= 255 

    return x

def extract_camera(ex):

    """Reformat camera properties into a dictionary"""

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

def extract_environment(ex):

    """Extract and decode environment features
    
    :note: time excluded (difficult to stack) >> ex['env_time']
    """

    weatherclasses = get_weather_classes(ex['env_wthr'])
    environment =  tf.stack([weatherclasses, 
                             tf.cast(ex['env_rain'], tf.int32), 
                             tf.cast(ex['env_snow'], tf.int32)])    
    return environment

def extract_cpos(ex, camera):

    """Extract vehicle pose information."""

    cpos = tf.stack([tf.sparse.to_dense(ex['pos_x']), 
                     tf.sparse.to_dense(ex['pos_y']), 
                     tf.sparse.to_dense(ex['pos_z']), 
                     tf.sparse.to_dense(ex['dim_x']), 
                     tf.sparse.to_dense(ex['dim_y']), 
                     tf.sparse.to_dense(ex['dim_z']), 
                     tf.sparse.to_dense(ex['rot_x']), 
                     tf.sparse.to_dense(ex['rot_y']), 
                     tf.sparse.to_dense(ex['rot_z'])], axis=1)
    return cpos

def extract_features(ex):

    """Extract and decode features"""

    vehicleclasses = get_vehicle_classes(ex['cls'])
    features = tf.stack([tf.cast(tf.sparse.to_dense(ex['dam']), tf.float32),
                         tf.sparse.to_dense(ex['occ']),
                         tf.sparse.to_dense(ex['tru']), 
                         tf.sparse.to_dense(vehicleclasses)], axis = 1)
    return features

def parse_detect_tfrecord(tfrecord, image_size):

    """Parse detection tfrecords. Extract full sized and reduced size images, 
    and camera properties.
    
    :param tfrecord: string tfrecord file path
    :param image_size: resize model image dimensions (if not None)

    :result: [image (full), camera] (if image_size = None)
             + [image (resize)]     (if image_size is specified)
    """

    ex = tf.io.parse_single_example(tfrecord, DETECT_FEATURE_MAP)

    im_full = tf.image.decode_jpeg(ex['image'], channels=3)    
    im_resize = tf.image.resize(im_full, (image_size, image_size))
    im_resize /= 255

    camera = extract_camera(ex)

    return {'camera': camera, 'image_full': im_full, 'input_image': im_resize}


def parse_model_tfrecord(tfrecord, image_size):
    
    """Parse individual tfrecords and extract image, target, camera and environment features

    :param tfrecord: string tfrecord file path
    :param image_size: resize target image dimensions

    :result: [image (reduced), cpos, features, camera, environment] 
    """
    
    ex = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)

    x = decode_and_resize_jpeg(ex['image'], image_size)
    camera = extract_camera(ex)

    environment = extract_environment(ex) # Row-wise: [weather class, rain, snow]

    N = tf.shape(ex['pos_x'])[0]

    def true_fn():
        cpos = extract_cpos(ex, camera) # Row-wise: [pos_x, pos_y, pos_z, dim_x, dim_y, dim_z, rot_x, rot_y, rot_z]
        features = extract_features(ex) # Row-wise: [dam, occ, tru, cls]
        return cpos, features

    def false_fn():       
        cpos = tf.zeros((0, 9))
        features = tf.zeros((0, 4))
        return cpos, features

    cpos, features = tf.cond(tf.greater(N, 0), 
                             true_fn,
                             false_fn)

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

    def true_fn():

        bb2d = get_bb2d(camera, cpos)
        feats = tf.transpose(features) 
        
        input_rpn = tf.stack([
        
            # Input: RPN [4 + 1 = 5]
        
            bb2d[0],      # xmin
            bb2d[1],      # ymin 
            bb2d[2],      # xmax
            bb2d[3],      # ymax                      
            feats[0]], # dam
                             axis = 1)

        screen = get_screen_center(camera, cpos)
    
        dims = cpos[:, 3:6] # [dimx, dimy, dimz]
        if cfg.USE_DIM_ANCHOR:                             
            med_dims = tf.constant(cfg.DIM_ANCHOR, tf.float32)
            dims -= med_dims    
        dims = tf.transpose(dims)

        euler = cpos[:, 6:]    
        quat = E2Q(euler)

        input_pose = tf.stack([

            # Input: Head [2 + 1 + 3 + 4 = 10]

            tf.cast(screen[0], tf.float32), # screenx 
            tf.cast(screen[1], tf.float32), # screeny
            cpos[:, 2],      # posz 
            dims[0],         # dimx
            dims[1],         # dimy
            dims[2],         # dimz
            quat[0],        # ix  
            quat[1],        # iy
            quat[2],        # iz
            quat[3]],       # iw
                              axis = 1)  
        
        occ, tru, cls = 1, 2, 3
        input_features = tf.stack([
            feats[occ],
            feats[tru], 
            feats[cls]], axis=1)

        return input_features, input_rpn, input_pose

    def false_fn():    

        input_features = tf.zeros((0, 3))
        input_rpn = tf.zeros((0, 5))
        input_pose = tf.zeros((0, 10))

        return input_features, input_rpn, input_pose

    input_features, input_rpn, input_pose = tf.cond(tf.greater(N, 0), 
                                                    true_fn, 
                                                    false_fn)
    
    paddings = [[0, cfg.MAX_GT_INSTANCES - N], [0, 0]] 

    padded_features = tf.pad(input_features, paddings)
    padded_pose = tf.pad(input_pose, paddings)
    padded_rpn = tf.pad(input_rpn, paddings)
    
    inputs = {'camera': camera,
              'environment': environment,
              'input_features': tf.ensure_shape(padded_features, (cfg.MAX_GT_INSTANCES, 3)),
              'input_image': x,              
              'input_pose': tf.ensure_shape(padded_pose, (cfg.MAX_GT_INSTANCES, 10)), 
              'input_rpn': tf.ensure_shape(padded_rpn, (cfg.MAX_GT_INSTANCES, 5))}            # seems alphabetical

    outputs = tf.constant(0)     

    return inputs, outputs

def load_ds(data_path, set, cfg):

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
                       
    ds = read_and_parse(data_path, set, cfg.IMAGE_SIZE)                               # Out: [x, cpos, features, camera, environment]
    istrain = (set == 'train')

    if istrain and cfg.ISAUGMENT: 
        augment_ds = ds.map(horizontal_flip)
        ds = ds.concatenate(augment_ds)

    ds = ds.map(lambda *x: transform_prebatch(*x, cfg))                               # Out: [x, padded input_gt, camera, environment] ADJUST
    
    if istrain: 
        ds = ds.shuffle(buffer_size=cfg.BUFFER_SIZE, reshuffle_each_iteration=True) 

    ds = ds.batch(cfg.BATCH_SIZE)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  
    return ds

def read_tfrecord(regex):

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
 
    ds = tf.data.Dataset.list_files(regex)    
    ds = tf.data.TFRecordDataset(ds)      
    return ds

def read_and_parse(data_path, set, image_size):
    
    regex = os.path.join(data_path, set) + "\\*.tfrecord"

    ds = read_tfrecord(regex)                                                         
    ds = ds.map(lambda e: parse_model_tfrecord(e, image_size))
    return ds

def load_all_ds(data_path, cfg):
    
    train_ds = load_ds(data_path, "train", cfg = cfg)
    val_ds = load_ds(data_path, "val", cfg = cfg)
    infer_ds = load_ds(data_path, "infer", cfg = cfg)
    
    return train_ds, val_ds, infer_ds


# SECTION 2: Detection ===============================================================

def load_detect_ds(regex, cfg):

    """Load tfrecord for detection

    :param regex: regex distinguishing tfrecord/s
    :param cfg: model configuration settings

    :result: {'camera', 'image_full', 'input_image'}     
    """

    ds = read_tfrecord(regex)
    ds = ds.map(lambda e: parse_detect_tfrecord(e, cfg.IMAGE_SIZE)) 
    ds = ds.batch(cfg.BATCH_SIZE)
    return ds   



