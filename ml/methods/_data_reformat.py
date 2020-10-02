"""
    Description: Supporting functions for reformatting data into a pre-yolo format
    Function: Format y and features  
    - PART A: 3D-to-2D projection, 2D bounding boxes
    - PART B: Screen co-ordinates, quarternions                
    - PART C: Vehicle class encoding 
"""

import tensorflow as tf
import math

pi = tf.constant(math.pi)
_false = tf.constant(False, dtype=tf.bool)
_true = tf.constant(True, dtype=tf.bool)

# PART A ==================================================================================

def compute_bb3d(camera, cpos_elem):
    
    """Computes 2D normalised screen co-ordinates of a vehicle's 3D bounding box vertices
    
    :param camera: camera property dictionary 
    :param cpos_elem: 9-element vector = [pos_x, pos_y, pos_z, 
                                          dim_x, dim_y, dim_z,
                                          rot_x, rot_y, rot_z]

    :return: (9, 2) array, where each row specifies (x, y) normalised coordinates    
    """

    # Target rotation
    
    cpos_elem_8 = get_rel_yaw(camera['R'], camera["yaw"], cpos_elem[8], _false)
    tar_cam_rot = tf.stack([cpos_elem[6], cpos_elem[7], cpos_elem_8])   
    R_tar = create_rot(tar_cam_rot)

    # Corner dimensions

    dims = cpos_elem[3:6]
    corner_dims = get_corner_dim(dims)
    corner_dims = tf.transpose(corner_dims)
    
    # Camera coordinates [Rotated corners + vehicle center]
    
    oriented_corners = tf.cast(tf.matmul(R_tar, corner_dims), tf.float64) 
    center = tf.cast(cpos_elem[:3], tf.float64)                           

    corner_cam = tf.transpose(tf.matmul(tf.transpose(camera['R']), oriented_corners)) + center     
    return tf.map_fn(lambda cw: C2S(camera, cw), corner_cam, dtype = tf.int16, parallel_iterations=8)    

@tf.function
def get_rel_yaw(cam_R, cam_yaw, tar_yaw, is_tar_to_cam):

    """Converts absolute gta-extracted vehicle yaw relative to the camera yaw, and vice versa    
    :note: camera zero-yaw axis is determined by the up/down orientation of its vertical axis
    :note: camera yaw is harmonised to ensure yaw is always defined relative to an upward-facing
           vertical axis

    :param cam_R: camera rotation matrix
    :param cam_yaw: absolute camera yaw
    :param tar_yaw: absolute vehicle yaw
    :param is_tar_to_cam: boolen specifying target-to-camera conversion direction

    :return: converted vehicle yaw
    """

    dir_yaw = tf.greater(cam_R[2, 1], 0)                          
    if is_tar_to_cam:            
        adj_yaw = realign_rad(tar_yaw - cam_yaw)
        if not dir_yaw: adj_yaw = realign_yaw(adj_yaw, _true)
    else: 
        if not dir_yaw: tar_yaw = realign_yaw(tar_yaw, _false)
        adj_yaw = tar_yaw + cam_yaw          
    return adj_yaw
 
def realign_yaw(theta, is_tar_to_cam):

    """Harmonises relative yaw measures by reorienting the camera's vertical axis upward (flip), 
    and vice versa, depending on the target-to-camera conversion direction

    :param theta: vehicle's relative yaw
    :param is_tar_to_cam: boolean specifying the target-to-camera conversion direction

    :result: float tensor 
    """

    f = tf.cond(is_tar_to_cam, lambda: pi, lambda: -pi)
    
    break_lb = tf.logical_and(tf.greater_equal(theta, -pi), tf.less_equal(theta, 0))    
    break_ub = tf.logical_and(tf.greater(theta, 0), tf.less_equal(theta, pi))
    
    theta = tf.case([(break_lb, lambda: theta + f),  # Test 1: theta >= -pi and theta <= 0 
                     (break_ub, lambda: theta - f)], # Test 2: theta > 0 and theta <= pi
                    exclusive=True)
    return theta                                                                   

def realign_rad(rad):

    """Simplifies radian values to their equivalent value between -pi and pi
    
    :param rad: input radian value

    :result: radian constant float tensor between -pi and pi 
    """

    within_b = tf.logical_and(tf.greater_equal(rad, -pi), tf.less_equal(rad, pi))
    break_lb = tf.logical_and(tf.less(rad, -pi), tf.greater_equal(rad, -2*pi))
    break_ub = tf.logical_and(tf.greater(rad, pi), tf.less_equal(rad, 2*pi))

    rad = tf.case([(within_b, lambda: rad), 
                   (break_lb, lambda: rad + 2*pi), 
                   (break_ub, lambda: rad - 2*pi)], 
                  exclusive = True)
    return rad


def create_rot(euler):

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

    Rx = tf.cast([[1, 0, 0],
                  [0, tf.cos(x), -tf.sin(x)],
                  [0, tf.sin(x), tf.cos(x)]], dtype=tf.float32)
    Ry = tf.cast([[tf.cos(y), 0, tf.sin(y)],
                  [0, 1, 0],
                  [-tf.sin(y), 0, tf.cos(y)]], dtype=tf.float32)
    Rz = tf.cast([[tf.cos(z), -tf.sin(z), 0],
                  [tf.sin(z), tf.cos(z), 0],
                  [0, 0, 1]], dtype=tf.float32)
    
    R = Rz * Rx * Ry
    return R

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

    ndc_x = camera['fx'] * cw[0]/cw[2] 
    ndc_y = camera['fy'] * cw[1]/cw[2]

    W, H = tf.cast(camera['w'], tf.float32), tf.cast(camera['h'], tf.float32)

    sw = tf.stack([(W - 1) * 0.5 *(1 + ndc_x), (H - 1) * 0.5 * (1 - ndc_y)])
    sw = tf.round(sw)
    sw = tf.cast(sw, tf.int16)
    return sw

@tf.function
def compute_bb2d(camera, cpos_elem):

    """Computes normalised, clipped 2D bounding box from a vehicle's projected 3D 
    bounding box vertices
    
    :param camera: camera property dictionary
    :param cpos_elem: 9-element vector = [pos_x, pos_y, pos_z, 
                                          dim_x, dim_y, dim_z,
                                          rot_x, rot_y, rot_z]

    :return: (xmin, ymin, xmax, ymax) normalised tensor
    """

    bb3d = compute_bb3d(camera, cpos_elem)
    x_vals = bb3d[:, 0]
    y_vals = bb3d[:, 1]

    xmin = tf.reduce_min(x_vals)
    xmax = tf.reduce_max(x_vals)
    ymin = tf.reduce_min(y_vals)
    ymax = tf.reduce_max(y_vals)

    W = tf.cast(camera['w'], dtype = tf.int16) 
    H = tf.cast(camera['h'], dtype = tf.int16) 

    if xmin < 0: xmin = tf.cast(0, tf.int16)
    if xmax > (W - 1): xmax = tf.cast(W - 1, tf.int16)
    if ymin < 0: ymin = tf.cast(0, tf.int16)
    if ymax > (H - 1): ymax = tf.cast(H - 1, tf.int16)

    xmin = tf.cast(xmin / (W - 1), tf.float32)
    xmax = tf.cast(xmax / (W - 1), tf.float32)
    ymin = tf.cast(ymin / (H - 1), tf.float32)
    ymax = tf.cast(ymax / (H - 1), tf.float32)

    return tf.stack([xmin, ymin, xmax, ymax])

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

    center = tf.map_fn(lambda x: C2S(camera, x[:3]), cpos, dtype = tf.int16)
    norm = tf.expand_dims(tf.stack([camera['w'] - 1, camera['h'] - 1]), axis=-1)  
    screen = tf.transpose(center) / tf.cast(norm, tf.int16) 
    return screen

def get_quart(cpos):

    """Group: Converts euler vehicle rotations to equivalent quarternion representations
    
    :param cpos: collated position, dimension and rotation information
                 for all detections in an image

    :result: (ndetections, [qx, qy, qz, qw])
    """
    
    euler = cpos[:, 6:]
    quart = tf.map_fn(euler_to_quart, euler, tf.float32)
    quart = tf.transpose(quart)    
    return quart

def euler_to_quart(euler):

    """Converts euler/tait-bryan angles to an equivalent quarternion representation
    :note: Z-X'-Y'' intrinsic rotation OR YXZ extrinsic rotation
    :note: rotation charcterised by unit quarternions
    
    :param euler: euler/tait-bryan angles corresponding to pitch, roll and yaw

    :result: (qx, qy, qz, qw)
    """
    
    hp = 0.5 * euler[0]
    hr = 0.5 * euler[1]
    hy = 0.5 * euler[2] 

    sinp, cosp = tf.sin(hp), tf.cos(hp)  
    sinr, cosr = tf.sin(hr), tf.cos(hr)
    siny, cosy = tf.sin(hy), tf.cos(hy)

    qx = cosr * sinp * cosy + sinr * cosp * siny
    qy = sinr * cosp * cosy - cosr * sinp * siny
    qz = cosr * cosp * siny - sinr * sinp * cosy
    qw = cosr * cosp * cosy + sinr * sinp * siny

    return tf.stack([qx, qy, qz, qw])

@tf.function
def quart_to_euler(q):

    """Converts a quarternion to an euler/tait-bryan angle representation
    :note: Z-X'-Y'' intrinsic rotation OR YXZ extrinsic rotation
    :note: extracts euler from a quarternion-generated rotation matrix

    :param q: (qx, qy, qz, qw) quarternion

    :result: euler (pitch, roll, yaw) representation
    """

    qx, qy, qz, qw = q[0], q[1], q[2], q[3]

    m13 = 2*(qx*qz + qy*qw)
    m21 = 2*(qx*qy + qz*qw)
    m22 = 1 - 2*(qx**2 + qz**2)
    m23 = 2*(qy*qz - qx*qw)
    m33 = 1 - 2*(qx**2 + qy**2)

    x = tf.asin(-m23) 
    
    if tf.math.less(tf.abs(m23), 0.9999999):
        y = tf.atan2(m13, m33)
        z = tf.atan2(m21, m22)
    else:
        y = tf.atan2(-m31, m11)
        z = tf.constant([0])

    return tf.stack([x, y, z])

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