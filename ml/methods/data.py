# Description: Read in, format and batch training and validation data
# Functions:
#    A. Read in tfrecord files
#    B. Reformat and transform targets
#    C. Build pipeline
#       > Batch
#       > Augmentations

import tensorflow as tf
import os

import sys
sys.path += ["C:\\Users\\Vaughn\\projects\\work\\drs\\ml\\methods\\", "C:\\Users\\Vaughn\\projects\\work\\drs\\ml\\"]

from config import cfg
from _data_reformat import get_bb2d, get_screen_center, get_quart, get_vehicle_classes
from _data_targets import transform_targets
from _data_augment import horizontal_flip

# A/B: Read in tfrecord and apply basic reformatting

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

    'cam_yaw': tf.io.FixedLenFeature([], tf.float32),
    'cam_R': tf.io.FixedLenFeature([], tf.string),
    'cam_w': tf.io.FixedLenFeature([], tf.int64),
    'cam_h': tf.io.FixedLenFeature([], tf.int64),
    'cam_fx': tf.io.FixedLenFeature([], tf.float32),
    'cam_fy': tf.io.FixedLenFeature([], tf.float32),

    'env_time': tf.io.FixedLenFeature([], tf.string), 
    'env_wthr': tf.io.FixedLenFeature([], tf.string),
    'env_rain': tf.io.FixedLenFeature([], tf.int64),
    'env_snow': tf.io.FixedLenFeature([], tf.int64)
    }

def parse_tfrecord(tfrecord, size):
    
    ex = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    
    x = tf.image.decode_jpeg(ex['image'], channels=3)
    x = tf.image.resize(x, (size, size))
    x /= 255 

    # Row-wise: Normal [pos_x, pos_y, pos_z, dim_x, dim_y, dim_z, rot_x, rot_y, rot_z]           

    cpos = tf.stack([tf.sparse.to_dense(ex['pos_x']), 
                     tf.sparse.to_dense(ex['pos_y']), 
                     tf.sparse.to_dense(ex['pos_z']), 
                     tf.sparse.to_dense(ex['dim_x']), 
                     tf.sparse.to_dense(ex['dim_y']), 
                     tf.sparse.to_dense(ex['dim_z']), 
                     tf.sparse.to_dense(ex['rot_x']), 
                     tf.sparse.to_dense(ex['rot_y']), 
                     tf.sparse.to_dense(ex['rot_z'])], axis=1)
    
    vehicleclasses = get_vehicle_classes(ex['cls'])
    features = tf.stack([tf.cast(tf.sparse.to_dense(ex['dam']), tf.float32),
                         tf.sparse.to_dense(ex['occ']),
                         tf.sparse.to_dense(ex['tru']), 
                         tf.sparse.to_dense(vehicleclasses)], axis = 1)

    cam_R = tf.io.decode_raw(ex['cam_R'], tf.float64)
    cam_R = tf.reshape(cam_R, (3, 3))
    camera = {'yaw': ex['cam_yaw'], 
              'R': cam_R, 
              'w': ex['cam_w'], 
              'h': ex['cam_h'],
              'fx': ex['cam_fx'], 
              'fy': ex['cam_fy']}

    environment = {"time": ex['env_time'], 
                   "wthr": ex['env_wthr'],
                   "rain": ex['env_rain'],
                   "snow": ex['env_snow']}

    return x, cpos, features, camera, environment

def transform_prebatch(x, cpos, features, camera, environment, max_boxes):

    # Output: [x, padded y, camera, environment]
    # Padded y: [xmin, ymin, xmax, ymax, dam status, (BASE)
    #            occ, tru, veh cls,                  (FEATURES)
    #            screenx, screeny, posz,             (HEAD)
    #            dimx, dimy, dimz,                   (HEAD)
    #            qx, qy, qz, qw]                     (HEAD)

    bb2d = get_bb2d(camera, cpos)
    xmin, ymin, xmax, ymax = 0, 1, 2, 3 
    
    dam, occ, tru, cls = 0, 1, 2, 3

    screen = get_screen_center(camera, cpos)
    posz, dimx, dimy, dimz = 2, 3, 4, 5

    quart = get_quart(cpos)
    ix, iy, iz, iw = 0, 1, 2, 3

    y = tf.stack([
                  # BASE
        
                  bb2d[xmin], 
                  bb2d[ymin], 
                  bb2d[xmax], 
                  bb2d[ymax],
                  features[:, dam], 
                  
                  # FEATURES
                  
                  features[:, occ],
                  features[:, tru],
                  features[:, cls],

                  # HEAD

                  tf.cast(screen[ix], tf.float32), 
                  tf.cast(screen[iy], tf.float32),
                  cpos[:, posz], 
                  cpos[:, dimx],
                  cpos[:, dimy],
                  cpos[:, dimz],
                  quart[ix],  
                  quart[iy], 
                  quart[iz], 
                  quart[iw]], axis=1)
    
    N = tf.gather(tf.shape(y), 0)
    paddings = [[0, max_boxes - N], [0, 0]] # Check padding
    y = tf.pad(y, paddings)

    return x, y # , camera, environment [ADJUST]
  
def transform_postbatch(x, y, cfg_mod, size): # camera, environment, [ADJUST]

    # y: ((grid_s, grid_m, grid_l), camera, environment)

    # Adapt y[:, :4]:
    #   Input: [N-batch, xmin, ymin, xmax, ymax]
    #  Output: ((N-batch, grid_s, grid_s, anchors_pscale, 4 + 1 + 1 + 3),
    #           (N-batch, grid_m, grid_m, anchors_pscale, 4 + 1 + 1 + 3),
    #           (N-batch, grid_l, grid_l, anchors_pscale, 4 + 1 + 1 + 3))
    # ... = 4 bb + obj + dam_prob + occ + tru + cls = 9

    yt = transform_targets(y, cfg_mod, size)
    return x, yt # , camera, environment [ADJUST]

# B/C: Further reformat, batch and shuffle the data

@tf.function
def load_ds(data_path, cfg_set, cfg_mod, size, batch_size, max_boxes, isaugment = False):

    ds = read_and_parse(data_path, size, istrain = cfg_set.ISTRAIN) 
    
    # Output: [x, cpos, features, camera, environment] 
      
    if isaugment:
        augment_ds = ds.map(horizontal_flip)
        ds = ds.concatenate(augment_ds)

    ds = ds.map(lambda x, cpos, features, camera, environment: 
                transform_prebatch(x, cpos, features, camera, environment, max_boxes))

    # Output: [x, y, camera, environment] 

    if cfg_set.ISTRAIN: ds.shuffle(buffer_size=cfg_set.BUFFER_SIZE, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)

    # > Batch mode

    ds = ds.map(lambda x, y: transform_postbatch(x, y, cfg_mod, size)) # , camera, environment [ADJUST]
    
    # Output: [x, y]    
       
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

def read_and_parse(data_path, size, istrain = True):

    # Output: [x, cpos, features, camera, environment]

    ds_set = "train" if istrain else "val"
    regex = os.path.join(data_path, ds_set) + "\\*.tfrecord"
    ds = tf.data.Dataset.list_files(regex)    
    
    # SHARDS: Here where distributed training (Not available currently)
    # INTERLEAVE: Where perform operations in parallel above parallel_calls (Possible training time improvement - see if needed later)
    # No CACHE as data does not fit into memory
    # Repeat does not seem necessary given reshuffle after each iteration

    ds = tf.data.TFRecordDataset(ds)
    ds = ds.map(lambda e: parse_tfrecord(e, size))    
    return ds

# TESTING
    
#data_path = "C:\\Users\\Vaughn\\projects\\work\\drs\\ml\\data\\"
#istrain = True
#cfg_mod = cfg.YOLO.V3
#size = 416
#batch_size = 4
#isaugment = True
#cfg_set = cfg.TRAIN
#max_boxes = 40


#x, cpos, features, camera, environment = list(ds.take(1))[0]
#tf.config.experimental_run_functions_eagerly(True)
# ex = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)


