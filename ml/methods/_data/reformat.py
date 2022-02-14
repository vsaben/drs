"""
    Description: Supporting functions for reformatting data into a pre-yolo format
    Function: Format y and features  
    - PART A: 3D-to-2D projection, 2D bounding boxes
    - PART B: Screen co-ordinates, quaternions            
    - PART C: Recollect annotations
    - PART D: Vehicle class encoding 
"""

import tensorflow as tf
import math

pi = tf.constant(math.pi)
_false = tf.constant(False , dtype=tf.bool)
_true = tf.constant(True, dtype=tf.bool)

# PART A ==================================================================================

def compute_bb3d_2d(camera, cpos_elem):

    """Converts bb3d in camera to screen space"""

    corner_cam = compute_bb3d_3d(camera, cpos_elem)
    return tf.map_fn(lambda cw: C2S(camera, cw), corner_cam, dtype = tf.int32, parallel_iterations=8)   

def compute_bb3d_3d(camera, cpos_elem):
    
    """Computes 2D normalised screen co-ordinates of a vehicle's 3D bounding box vertices
    
    :param camera: camera property dictionary 
    :param cpos_elem: 9-element vector = [pos_x, pos_y, pos_z, 
                                          dim_x, dim_y, dim_z,
                                          rot_x, rot_y, rot_z]

    :return: (8, 3) array, where each row specifies (x, y, z) camera coordinates    
    """

    # Target rotation
    
    cpos_elem_8 = get_abs_tar_yaw(camera, cpos_elem[8])
    tar_abs_rot = tf.stack([cpos_elem[6], cpos_elem[7], cpos_elem_8])   
    R_tar = create_tar_rot(tar_abs_rot)

    # Corner dimensions

    dims = cpos_elem[3:6]
    corner_dims = get_corner_dim(dims)
    corner_dims = tf.transpose(corner_dims)
    
    # Camera coordinates [Rotated corners + vehicle center]
    
    oriented_corners = tf.cast(tf.matmul(R_tar, corner_dims), tf.float64)
    center = tf.cast(cpos_elem[:3], tf.float64)

    return tf.transpose(tf.matmul(tf.transpose(tf.cast(camera['R'], tf.float64)), 
                                  oriented_corners)) + center
     

def get_abs_tar_yaw(camera, rel_yaw):

    """Converts derived relative yaw to absolute gta-extracted vehicle yaw.    
    :note: camera zero-yaw axis is determined by the up/down orientation of its vertical axis
    :note: camera yaw is harmonised to ensure yaw is always defined relative to an upward-facing
           vertical axis

    :param camera: camera property dictionary
    :param rel_yaw: relative vehicle-to-camera yaw

    :return: absolute vehicle yaw (as specificied in GTA)
    """

    cam_yaw = tf.cast(camera['rot'][2], tf.float32)
    dir_yaw = tf.greater(camera['R'][2, 1], 0)   

    rel_yaw = realign_rad(rel_yaw)

    r_yaw = tf.cond(tf.math.logical_not(dir_yaw), 
                    lambda: realign_rel_yaw(rel_yaw), 
                    lambda: rel_yaw)
    return r_yaw + cam_yaw
 
def realign_rel_yaw(rel_yaw):

    """Regenerates gta-extracted target yaw by undoing camera-target vertical axis reorientation

    :param theta: vehicle's relative yaw
    """
    
    break_lb = tf.logical_and(tf.greater_equal(rel_yaw, -pi), tf.less_equal(rel_yaw, 0))    
    break_ub = tf.logical_and(tf.greater(rel_yaw, 0), tf.less_equal(rel_yaw, pi))   

    rel_yaw = tf.case([(break_lb, lambda: rel_yaw - pi),  # Test 1: theta >= -pi and theta <= 0 
                       (break_ub, lambda: rel_yaw + pi)], # Test 2: theta > 0 and theta <= pi
                      default=lambda: 0.0, # Test-stage
                      exclusive=True)
    return rel_yaw                                                                   

def realign_rad(rad):

    """Simplifies radian values to their equivalent value between -pi and pi
    
    :note: already specified with rotation axis
    :note: tf.mod >> Not same as python (remainder is not the same sign as the denominator)

    :param rad: input radian value

    :result: radian constant float tensor between -pi and pi 
    """

    npi = tf.math.floordiv(tf.abs(rad), pi) # x // y 
    remainder = tf.math.mod(rad, pi)

    ispos = tf.equal(remainder / tf.abs(remainder), 1)
    iseven = tf.equal(tf.math.mod(npi, 2), 0) # even (-pi, 0); odd (0, pi)
    
    pos_even = tf.logical_and(ispos, iseven)
    pos_odd = tf.logical_and(ispos, tf.logical_not(iseven))
    neg_even = tf.logical_and(tf.logical_not(ispos), iseven)
    neg_odd = tf.logical_and(tf.logical_not(ispos), tf.logical_not(iseven))
    
    res_rad = tf.case([(pos_even, lambda: remainder), 
                       (pos_odd, lambda: -pi + remainder), 
                       (neg_even, lambda: remainder), 
                       (neg_odd, lambda: pi + remainder)], 
                      exclusive = True)

    return res_rad


    # within_b = tf.logical_and(tf.greater_equal(rad, -pi), tf.less_equal(rad, pi))
    # break_lb = tf.logical_and(tf.less(rad, -pi), tf.greater_equal(rad, -2*pi))
    # break_ub = tf.logical_and(tf.greater(rad, pi), tf.less_equal(rad, 2*pi))

    # rad = tf.case([(within_b, lambda: rad), 
    #                (break_lb, lambda: rad + 2*pi), 
    #                (break_ub, lambda: rad - 2*pi)], 
    #               exclusive = True)
    # return rad

def create_tar_rot(euler):

    """Creates a rotation matrix from euler/tait-bryan angles
    :note: Z-X'-Y'' intrinsic rotation OR YXZ extrinsic rotation
    :note: right-multiplication of the transposed rotation 
           matrix generates the required rotation order

    :param euler: euler/tait-bryan angles
                  x = pitch (pi < x < 0) 
                  y = roll (-pi < y < pi)
                  z = yaw (-pi < z < pi)
    
    :result: (3, 3) rotation matrix
    """

    x, y, z = euler[0], euler[1], euler[2]

    Rx = tf.stack([[1, 0, 0],
                   [0, tf.cos(x), -tf.sin(x)],
                   [0, tf.sin(x), tf.cos(x)]])
    Ry = tf.stack([[tf.cos(y), 0, tf.sin(y)],
                   [0, 1, 0],
                   [-tf.sin(y), 0, tf.cos(y)]])
    Rz = tf.stack([[tf.cos(z), -tf.sin(z), 0],
                   [tf.sin(z), tf.cos(z), 0],
                   [0, 0, 1]])
    
    return tf.matmul(Rz, tf.matmul(Rx, Ry))        

def create_cam_rot(euler):

    I_switch = tf.constant([[1, 0, 0], 
                            [0, 0, 1], 
                            [0, 1, 0]], dtype=tf.float64)
    R = create_tar_rot(euler)    
    return tf.matmul(R, I_switch)

DIM_KEYS = ["ftl", "ftr", "fbl", "fbr",   
            "btl", "btr", "bbl", "bbr"]

def get_corner_dim(dims):  
    
    """Calculates each 3D bounding box vertex's center offset

    :param dims: half-width (x), length (y) and height (z) dimensions from
                 a vehicle's center point

    :return: (8, 3) array, where each row specifies a different vertex's center offset 
    """

    x, y, z = dims[0], dims[1], dims[2]
    corner_dims = tf.cast([[-x, y, z],     # ftl  
                           [x, y, z],      # ftr    
                           [-x, y, -z],    # fbl  
                           [x, y, -z],     # fbr   
                           [-x, -y, z],    # btl 
                           [x, -y, z],     # btr   
                           [-x, -y, -z],   # bbl 
                           [x, -y, -z]],   # bbr
                           dtype=tf.float32)    
    return corner_dims

def C2S(camera, cw):

    """Converts co-ordinates in camera space to screen space

    :param camera: camera property dictionary
    :param cw: co-ordinate in camera space

    :return: (x, y) integer screen co-ordinate tensor
    """

    cw = tf.cast(cw, tf.float32)

    ndcx = camera['fx'] * cw[0]/cw[2] 
    ndcy = camera['fy'] * cw[1]/cw[2]

    W, H = tf.cast(camera['w'], tf.float32), tf.cast(camera['h'], tf.float32)

    sw = tf.stack([(W - 1) * 0.5 *(1 + ndcx), (H - 1) * 0.5 * (1 - ndcy)])
    sw = tf.round(sw)
    sw = tf.cast(sw, tf.int32)
    return sw

def compute_bb2d(camera, cpos_elem):

    """Computes normalised, clipped 2D bounding box from a vehicle's projected 3D 
    bounding box vertices
    
    :param camera: camera property dictionary
    :param cpos_elem: 9-element vector = [pos_x, pos_y, pos_z, 
                                          dim_x, dim_y, dim_z,
                                          rot_x, rot_y, rot_z]

    :return: (xmin, ymin, xmax, ymax) normalised tensor
    """

    bb3d = compute_bb3d_2d(camera, cpos_elem)
    x_vals = bb3d[:, 0]
    y_vals = bb3d[:, 1]

    W = tf.cast(camera['w'], tf.int32) 
    H = tf.cast(camera['h'], tf.int32) 

    xmin = tf.clip_by_value(tf.reduce_min(x_vals), 0, W - 1) / (W - 1) # float32
    xmax = tf.clip_by_value(tf.reduce_max(x_vals), 0, W - 1) / (W - 1)
    ymin = tf.clip_by_value(tf.reduce_min(y_vals), 0, H - 1) / (H - 1)
    ymax = tf.clip_by_value(tf.reduce_max(y_vals), 0, H - 1) / (H - 1)

    return tf.cast(tf.stack([xmin, ymin, xmax, ymax]), tf.float32)

def get_bb2d(camera, cpos):

    """Group: Calculates all vehicles wide 2D bounding box
    
    :note: a wide 2D bbox encapsulates projected 3D bbox vertices

    :param camera: camera property dictionary
    :param cpos: collated position, dimension and rotation information
                 for all detections in an image

    :result: (ndetections, [xmin, ymin, xmax, ymax]) normalised 2D co-ordinates
    """

    bb2d = tf.map_fn(lambda x: compute_bb2d(camera, x), cpos, dtype = tf.float32)
    return tf.transpose(bb2d)

# PART B =========================================================================

def get_screen_center(camera, cpos):
    
    """Group: Calculates all vehicle center points within an image

    :param camera: camera property dictionary
    :param cpos: collated position, dimension and rotation information
                 for all detections in an image

    :return: (ndetections, [x, y]) normalised 2D co-ordinates
    """

    center = tf.map_fn(lambda x: C2S(camera, x[:3]), cpos, dtype = tf.int32)
    norm = tf.expand_dims(tf.stack([camera['w'] - 1, camera['h'] - 1]), axis=-1)  
    screen = tf.transpose(center) / tf.cast(norm, tf.int32) 
    return screen

def get_quat(cpos):

    """Group: Converts euler vehicle rotations to equivalent quaternion representations
    
    :param cpos: collated position, dimension and rotation information
                 for all detections in an image

    :result: (ndetections, [qx, qy, qz, qw])
    """
    
    euler = cpos[:, 6:]
    q = tf.map_fn(euler_to_quat, euler, tf.float32)
    q = tf.transpose(q)    
    return q

def E2Q(euler):

    """Converts euler/tait-bryan angles to an equivalent quaternion representation
    :note: YXZ extrinsic rotation
    :note: rotation characterised by unit quaternions
    :note: source https://github.com/mrdoob/three.js/blob/master/src/math/Quaternion.js

    :param euler: euler/tait-bryan angles corresponding to pitch, roll and yaw

    :result: (qx, qy, qz, qw)
    """
    
    p, r, y = tf.split(euler, 3, axis = -1)

    hp = 0.5 * p
    hr = 0.5 * r
    hy = 0.5 * y 

    sinp, cosp = tf.sin(hp), tf.cos(hp)  
    sinr, cosr = tf.sin(hr), tf.cos(hr)
    siny, cosy = tf.sin(hy), tf.cos(hy)

    qx = cosr * sinp * cosy - sinr * cosp * siny
    qy = sinr * cosp * cosy + cosr * sinp * siny
    qz = sinr * sinp * cosy + cosr * cosp * siny
    qw = cosr * cosp * cosy - sinr * sinp * siny

    q = tf.concat([qx, qy, qz, qw], axis = 1)
    return tf.transpose(q)

def Q2E(q):

    """Converts a quaternion to an euler/tait-bryan angle representation
    :note: YXZ extrinsic rotation
    :note: extracts euler from a quaternion-generated rotation matrix
    :note: source https://github.com/mrdoob/three.js/blob/master/src/math/Euler.js

    :param q: (qx, qy, qz, qw) quaternion

    :result: euler (pitch, roll, yaw) representation
    """

    q /= tf.norm(q, axis = -1, keepdims = True)  
    qx, qy, qz, qw = tf.split(q, 4, axis = -1)

    m11 = qw**2 + qx**2 - qy**2 - qz**2
    m12 = 2*(qx*qy - qw*qz)
    m21 = 2*(qw*qz + qx*qy)
    m22 = qw**2 - qx**2 + qy**2 - qz**2
    m31 = 2*(qx*qz - qw*qy)
    m32 = 2*(qw*qx + qy*qz)
    m33 = qw**2 - qx**2 - qy**2 + qz**2
    
    x = tf.asin(tf.clip_by_value(m32, 
                                 clip_value_min=-1, 
                                 clip_value_max=1)) 
    
    cond = tf.cast(tf.math.less(tf.abs(m32), 0.9999999), tf.float32)

    y = cond * tf.atan2(-m31, m33) + (1 - cond) * tf.constant([0], tf.float32)
    z = cond * tf.atan2(-m12, m22) + (1 - cond) * tf.atan2(m21, m11)
    
    return tf.concat([x, y, z], axis = -1)

def out_pose_to_cpos(cameras, pose, cfg, stack = False):

    """Transforms raw, batched camera and pose 
    information into a list cpos format (refer result)

    :param cameras: unbatched camera dictionaries
    :param pose: [ndetect, 
                    [center (2) {screen}, 
                     depth (1) {absolute/relative}, 
                     dims (3) {half},
                     quat (4)]

    :result: [[n_roi, [pos (3), dims (3), rot (3)]], ...]
              per image
    """

    nbatch = len(pose)
    cpos_lst = [pose_to_cpos(pose[b], cameras[b], cfg) for b in tf.range(nbatch)]

    if not stack:
        return cpos_lst

    cpos = [c for c in cpos_lst if c != []]
    return tf.concat(cpos, axis=0, name='out_cpos')

def unpad(tensor, cond_idx):

    """Remove zero padding from a tensor"""

    cond = tensor[..., cond_idx]  
    pos_ix = tf.where(cond > 0)            # [none, 5/6] --> (instance index, cond true index)
    
    return tf.gather_nd(tensor, pos_ix)

def pose_to_cpos(pose, camera, cfg):

    """Converts predicted pose output to an annotation format

    :param pose: [detect, 10] 
         <center (2) {normalised to image} + 
          depth (1) + 
          dims (3) {absolute/relative} + 
          quat (4) = 10>
    :param camera: camera property dictionary

    :result: pose annotations in their original format
             {ndetect, pos (3) + dims (3) + rot (3)}
    """

    ndetect = tf.shape(pose)[0]
    if ndetect == 0:
        return [] 

    # 1-3: Position

    pos = out_pos_to_cam(pose, camera)

    # 4-6: Dimensions 

    dim = pose[..., 3:6]
    if cfg['USE_DIM_ANCHOR']:
        med_dims = tf.constant(cfg['DIM_ANCHOR'], tf.float32)
        dim += med_dims    

    dim = tf.nn.relu(dim)

    # 7-9: Rotation

    quat = pose[..., -4:]
    rot = Q2E(quat)

    return tf.concat([pos, dim, rot], axis=-1)


def out_pos_to_cam(pose, camera):

    """Screen to camera coordinates

    :param center: normalised center coordinates
    :param Z: predicted depth values
    :param cam: [W (1) + H (1) + fx (1) + fy (1) = 4]

    :result: [X, Y, Z] in camera coordinates 
    """

    if len(tf.shape(pose)) == 1:
        pose = tf.expand_dims(pose, axis=0)

    center = pose[..., :2] 
    Z = tf.expand_dims(pose[:, 2], axis=1)

    W = tf.cast(tf.squeeze(camera['w']), tf.float32)
    H = tf.cast(tf.squeeze(camera['h']), tf.float32)
    fx = camera['fx']
    fy = camera['fy'] 

    WH = tf.stack([W, H]) - 1
    x, y = tf.split(center * WH, 2, axis = -1)

    X = ((2*x)/(W - 1) - 1)*(Z/fx)
    Y = (1 - (2*y)/(H - 1))*(Z/fy)

    return tf.concat([X, Y, Z], axis = -1)

def unbatch_dict(batched_dict, nbatch = -1):

    """Unbatches amalgamated dictionaries"""
  
    if nbatch == -1:
        nbatch = list(batched_dict.values())[0].shape[0]

    res = [{k: v[i] for k, v in batched_dict.items()} 
            for i in range(nbatch)]
    return res
# PART C ======================================================================== 

WEATHER_CLASSES = {0:'Blizzard', 
                   1:'Christmas', 
                   2:'Clear', 
                   3:'Clearing', 
                   4:'Clouds', 
                   5:'Extra Sunny', 
                   6:'Foggy', 
                   7:'Haloween', 
                   8:'Neutral', 
                   9:'Overcast', 
                   10:'Raining', 
                   11:'Smog', 
                   12:'Snowing', 
                   13:'Snowlight', 
                   14:'Thunderstorm',
                   15:'Other'}

weather_table = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(list(WEATHER_CLASSES.values())),
       values=tf.range(len(WEATHER_CLASSES), dtype=tf.int32),
    ),
    default_value=tf.constant(len(WEATHER_CLASSES) - 1, tf.int32),
    name="weather_class"
)

def get_weather_classes(ex_cls):
    """Encodes weather class with an assigned, unique integer 
    :param ex_cls: weather class string    
    :result: assigned weather class integer
    """
    return weather_table.lookup(ex_cls)


VEHICLE_CLASSES = {0:'Commercial',
                   1:'Compacts', 
                   2:'Coupes',
                   3:'Cycles', 
                   4:'Emergency', 
                   5:'Industrial', 
                   6:'Motorcycles', 
                   7:'Muscle', 
                   8:'Offroad',
                   9:'Sedans', 
                   10:'Service', 
                   11:'Sports', 
                   12:'SportsClassics', 
                   13:'Super', 
                   14:'SUVs', 
                   15:'Utility', 
                   16:'Vans', 
                   17: 'Other'}

vehicle_table = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(list(VEHICLE_CLASSES.values())),
       values=tf.range(len(VEHICLE_CLASSES), dtype=tf.float32),
    ),
    default_value=tf.constant(len(VEHICLE_CLASSES) - 1, tf.float32),
    name="vehicle_class"
)

def get_vehicle_classes(ex_cls):
    """Encodes vehicle class with an assigned, unique integer 
    :param ex_cls: vehicle class string    
    :result: assigned vehicle class integer
    """
    return vehicle_table.lookup(ex_cls)