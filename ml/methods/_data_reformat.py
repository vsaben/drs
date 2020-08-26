# Description: Supporting functions for reformatting data into a pre-yolo format
# Function: 

#  [1] Reformat y into y1, y2 

#   A: y1 = [xmin, ymin, xmax, ymax, dam_int]
#   B: y2 = [screen_x, screen_y, pos_Z, 
#            dim_x, dim_y, dim_z, 
#            quart_x, quart_y, quart_z, quart_w]    

#  [2] Reformat features: vehicle class 

import tensorflow as tf

import math
pi = tf.constant(math.pi)

_false = tf.constant(False, dtype=tf.bool)
_true = tf.constant(True, dtype=tf.bool)

# 1A: Y1 ------------------------------------------------------------------------------

def compute_bb3d(camera, cpos_elem):
        
    # > Target rotation
    
    cpos_elem_8 = get_rel_yaw(camera['R'], camera["yaw"], cpos_elem[8], _false)
    tar_cam_rot = tf.stack([cpos_elem[6], cpos_elem[7], cpos_elem_8])   
    R_tar = create_rot(tar_cam_rot)

    # > Corner dimensions

    dims = cpos_elem[3:6]
    corner_dims = get_corner_dim(dims)
    corner_dims = tf.transpose(corner_dims)
    
    # > Camera coordinates [Rotated corners + vehicle center]
    
    oriented_corners = tf.cast(tf.matmul(R_tar, corner_dims), tf.float64) 
    center = tf.cast(cpos_elem[:3], tf.float64)                           

    corner_cam = tf.transpose(tf.matmul(tf.transpose(camera['R']), oriented_corners)) + center     
    return tf.map_fn(lambda cw: C2S(camera, cw), corner_cam, dtype = tf.int16, parallel_iterations=8)    


# i: Target rotation

@tf.function
def get_rel_yaw(cam_R, cam_yaw, tar_yaw, is_tar_to_cam):
    dir_yaw = tf.greater(cam_R[2, 1], 0)                          
    if is_tar_to_cam:            
        adj_yaw = realign_rad(tar_yaw - cam_yaw)
        if not dir_yaw: adj_yaw = realign_yaw(adj_yaw, _true)
    else: 
        if not dir_yaw: tar_yaw = realign_yaw(tar_yaw, _false)
        adj_yaw = tar_yaw + cam_yaw          
    return adj_yaw
 
def realign_yaw(theta, is_tar_to_cam):

    f = tf.cond(is_tar_to_cam, lambda: pi, lambda: -pi)
    
    break_lb = tf.logical_and(tf.greater_equal(theta, -pi), tf.less_equal(theta, 0))    
    break_ub = tf.logical_and(tf.greater(theta, 0), tf.less_equal(theta, pi))
    
    theta = tf.case([(break_lb, lambda: theta + f),  # Test 1: theta >= -pi and theta <= 0 
                     (break_ub, lambda: theta - f)], # Test 2: theta > 0 and theta <= pi
                    exclusive=True)
    return theta                                                                   

def realign_rad(rad):
    within_b = tf.logical_and(tf.greater_equal(rad, -pi), tf.less_equal(rad, pi))
    break_lb = tf.logical_and(tf.less(rad, -pi), tf.greater_equal(rad, -2*pi))
    break_ub = tf.logical_and(tf.greater(rad, pi), tf.less_equal(rad, 2*pi))

    rad = tf.case([(within_b, lambda: rad), 
                   (break_lb, lambda: rad + 2*pi), 
                   (break_ub, lambda: rad - 2*pi)], 
                  exclusive = True)
    return rad


def create_rot(euler):

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

# ii: Corner dimensions 

def get_corner_dim(dims):     
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


# iii: Camera co-ordinates

def C2S(camera, cw):

    cw = tf.cast(cw, tf.float32)

    ndc_x = camera['fx'] * cw[0]/cw[2] 
    ndc_y = camera['fy'] * cw[1]/cw[2]

    W, H = tf.cast(camera['w'], tf.float32), tf.cast(camera['h'], tf.float32)

    sw = tf.stack([(W - 1) * 0.5 *(1 + ndc_x), (H - 1) * 0.5 * (1 - ndc_y)])
    sw = tf.round(sw)
    sw = tf.cast(sw, tf.int16)
    return sw

# iv: Get 2D BB

@tf.function
def compute_bb2d(camera, cpos_elem):

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

    # Output: xmin, ymin, xmax, ymax (row-wise)

    bb2d = tf.map_fn(lambda x: compute_bb2d(camera, x), cpos, dtype = tf.float32)
    return tf.transpose(bb2d)

# 1B: Y2 ----------------------------------------------------------------------------------

def get_screen_center(camera, cpos):
    
    # Output: standardised screen_x, screen_y (row-wise)

    center = tf.map_fn(lambda x: C2S(camera, x[:3]), cpos, dtype = tf.int16)
    norm = tf.expand_dims(tf.stack([camera['w'] - 1, camera['h'] - 1]), axis=-1)  
    screen = tf.transpose(center) / tf.cast(norm, tf.int16) 
    return screen

def get_quart(cpos):

    # Output: quart_x, quart_y, quart_z, quart_w (row-wise)

    euler = cpos[:, 6:]
    quart = tf.map_fn(euler_to_quart, euler, tf.float32)
    quart = tf.transpose(quart)    
    return quart

def euler_to_quart(euler):

    # Assume: Z-X'-Y'' intrinsic rotation <> YXZ extrinsic rotation
    # Rotation: pitch [pi < x < 0], roll [-pi < y < pi], yaw [-pi < z < pi]
    #           Characterised by unit quaternion
    

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

    # Creates R and extracts corresponding euler
    # Convention: YXZ [Refer NASA, three.js]

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

# 2A: Features --------------------------------------------------------------

#VEHICLE_CLASSES = {0:'Commercial',
#                   1:'Compacts', 
#                   2:'Coupes',
#                   3:'Cycles', 
#                   4:'Emergency', 
#                   5:'Industrial', 
#                   6:'Motorcycles', 
#                   7:'Muscle', 
#                   8:'Offroad',
#                   9:'Sedans', 
#                   10:'Service', 
#                   11:'Sports', 
#                   12:'SportsClassics', 
#                   13:'Super', 
#                   14:'SUVs', 
#                   15:'Utility', 
#                   16:'Vans'}

vehicle_table = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(['Commercial',
                          'Compacts', 
                          'Coupes',
                          'Cycles', 
                          'Emergency', 
                          'Industrial', 
                          'Motorcycles', 
                          'Muscle', 
                          'Offroad',
                          'Sedans', 
                          'Service', 
                          'Sports', 
                          'SportsClassics', 
                          'Super', 
                          'SUVs', 
                          'Utility', 
                          'Vans']),
       values=tf.range(0, 17, dtype=tf.float32),
    ),
    default_value=tf.constant(-1, tf.float32),
    name="vehicle_class"
)

def get_vehicle_classes(ex_cls):
    return vehicle_table.lookup(ex_cls)