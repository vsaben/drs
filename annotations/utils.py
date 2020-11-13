"""
    Description: Supporting world-to-screen projection functions
    Function:
    - PART A: Camera (class)
    - PART B: Dimensions
    - PART C: Rotation
    - PART D: Projection
"""

import numpy as np
import utils

# PART A: Camera ============================================================

class Camera(object):

    """Collects and formats camera properties, as required in PARTS B and C.
    
    :note: C is a camera calibration matrix. It combines camera direction and
    projection elements (V*P). These components are separated to allow for 
    calculations with respect to camera co-ordinates.
    
    :result fx, fy: clip-to-ndc projection factors
    """

    def __init__(self, camera):

        self.pos = np.array(camera["location"], dtype=np.float64)
        self.rot = np.radians(camera["rotation"], dtype=np.float64)
        self.R = utils.create_camera_rot(self.rot)
        
        #C = np.array(camera["C"]["Values"], dtype=np.float64).reshape((4, 4)).T
        #R_raw = C[1:, :3]
        #R_norm = [x/np.linalg.norm(x) for x in R_raw]
        #self.R = np.vstack(R_norm).T        
               
        self.w = camera["screenW"]
        self.h = camera["screenH"]

        self.vfov = np.radians(camera["vfov"], dtype=np.float64)        
        self.fy = 1/np.tan(self.vfov/2)
        self.fx = self.h/self.w * self.fy
 
# PART B: Dimensions ===========================================================

def get_directional_vectors(position):
    
    """Gets right (r), forward (f) and up (u) directional vectors"""
 
    r = np.array(position["right_vec"], dtype=np.float64)
    f = np.array(position["forward_vec"], dtype=np.float64) 
    u = np.array(position["up_vec"], dtype=np.float64) 
    return r, f, u
   
def get_balanced_center(position):

    """Recalculates target center (in world coordinates), at the point where 
    each half-dimension on either side any axes are equal"""

    W = np.array(position["location"], dtype=np.float64)

    r, f, u = get_directional_vectors(position)
    bbox_min = np.array(position["bbox_min"], dtype=np.float64)
    bbox_max = np.array(position["bbox_max"], dtype=np.float64)
        
    x_off, y_off, z_off = bbox_min + bbox_max
    return W + r * x_off + f * y_off + u * z_off 

def get_half_dims(position):

    """Gets target half (x, y, z) dimensions (in meters)"""
    
    bbox_min = np.array(position["bbox_min"], dtype=np.float64)
    bbox_max = np.array(position["bbox_max"], dtype=np.float64)   
    return (bbox_max - bbox_min) / 2
          
def get_corner_dim(dims): 
    
    """Target corner offset dictionary. Key naming convention

    |f/b| = front/back
    |t/b| = top/bottom
    |l/r| = left/right    
    
    """

    x, y, z = dims
    corner_dims = {"ftl": np.array([-x, y, z], dtype=np.float64),  
                   "ftr": np.array([x, y, z], dtype=np.float64),    
                   "fbl": np.array([-x, y, -z], dtype=np.float64),  
                   "fbr": np.array([x, y, -z], dtype=np.float64),   
                   "btl": np.array([-x, -y, z], dtype=np.float64),  
                   "btr": np.array([x, -y, z], dtype=np.float64),   
                   "bbl": np.array([-x, -y, -z], dtype=np.float64), 
                   "bbr": np.array([x, -y, -z], dtype=np.float64)}  
    return corner_dims

# PART C: Rotation =============================================================

def create_target_rot(euler):

    """Creates target rotation matrix
    
    :note: not transposed in application

    :param euler: euler angles corresponding to rotations about the 
                  x-axis (pitch/right)
                  y-axis (roll/forward)
                  z-axis (yaw/up)

    :result: [right|forward|up] column-wise 3x3 matrix

    """

    x, y, z = euler 

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]], dtype=np.float64)
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]], dtype=np.float64)
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]], dtype=np.float64)
    
    return Rz @ Rx @ Ry 

def create_camera_rot(euler):

    """Creates camera rotation matrix. Swaps forward and up 
    directional vectors.
    
    :note: transposed in application

    :param euler: x-axis *(pitch/right)
                  y-axis *(roll/forward)
                  z-axis *(yaw/up)

    :result: [right|forward|up] column-wise 3x3 matrix
    """

    I_switch = np.array([[1, 0, 0], 
                         [0, 0, 1], 
                         [0, 1, 0]])

    R_zxy = create_target_rot(euler)
    R_cam = R_zxy @ I_switch

    return R_cam 

def get_rel_yaw(cam_yaw, tar_yaw, is_pos_axis = True, is_rel_cam = True):
    
    """Calculates the target's yaw relative to that of the camera
    
    :param cam_yaw: absolute camera yaw (in radians)
    :param tar_yaw: absolute target yaw (in radians)
    :option is_pos_axis: direction of perpendicular forward axis
                         check: R_cam[2, 1] > 0
    :option is_rel_cam: (False) absolute <> rel yaw (True)

    :result: yaw value as defined by options
    """

    if is_rel_cam:
        rel_yaw = realign_rad(tar_yaw - cam_yaw)
        if not is_pos_axis: 
            rel_yaw = realign_yaw(rel_yaw, True)
        return rel_yaw
    else:
        if not is_pos_axis: 
            tar_yaw = realign_yaw(tar_yaw, False)
        tar_yaw += cam_yaw
        return tar_yaw
    
def realign_rad(rad):

    """Ensures radian e [-pi, pi]"""
      
    if rad >= -np.pi and rad <= np.pi: return rad
    if rad < -np.pi and rad >= -2*np.pi: return rad + 2*np.pi
    if rad > np.pi and rad <= 2*np.pi : return rad - 2*np.pi
 
def realign_yaw(theta, is_rel_cam):

    """Inverts forward perpendicular axis direction relative 
    to the camera matrix (if needed)
    """

    f = np.pi if is_rel_cam else -np.pi 
    
    if theta >= -np.pi and theta <= 0: return theta + f
    if theta > 0 and theta <= np.pi: return theta - f


"""
def euler_from_R(R, rot_seq = 'XZY'):

    Extracts euler angles from the rotation matrix | rotation sequence order. 
    Calculates target rotations relative to the camera.

    :param R: rotation matrix with rot_seq order 
    :option rot_seq: rotation sequence order
                     only default is used
                     additional calculated during testing 

    :result: euler angles corresponding to create_rot (in radians)
    

    if rot_seq == 'XZY':

        z = np.asin(-R[0, 1])
        cz = np.cos(z)

        y = np.acos(R[0, 0]/cz)
        x = np.acos(R[1, 1]/cz)

    if rot_seq == 'XYZ':

        y = np.asin(R[0, 2])
        cy = np.cos(y)

        x = np.acos(R[2, 2]/cy)
        z = np.acos(R[0, 0]/cy)

    if rot_seq == 'ZYX':

        y = np.asin(-R[2, 0])
        cy = np.cos(y)

        x = np.asin(R[2, 1]/cy)
        z = np.acos(R[0, 0]/cy)

    if rot_seq == 'YXZ':

        x = np.asin(-R[1, 2])
        cx = np.cos(x)

        y = np.acos(R[2, 2]/cx) 
        z = np.asin(R[1, 0]/cx)

    if rot_seq == 'ZXY': # [-R[:, 0]]

        x = np.asin(R[2, 1])
        cx = np.cos(x)

        y = np.acos(R[2, 2]/cx) 
        z = np.asin(-R[0, 1]/cx)

    return np.array([x, y, z])
"""

# PART D: Projection =============================================================

def W2C(camera, W):

    """World-to-camera projection"""

    return np.matmul(camera.R.T, W - camera.pos)

def C2S(camera, cW):

    """Camera-to-screen projection"""

    ndc_x = camera.fx * cW[0]/cW[2] 
    ndc_y = camera.fy * cW[1]/cW[2]

    # ndc_x e [-1, 1]: 
    #    ] -1 --> 0 
    #    ] 0  --> screen_w / 2
    #    ] 1  --> screen_w

    # BUT image array e [0, screen_w - 1]
    # THEREFORE remap [0, screen_w] to [0, screen_w -1]
    # BY x * (screen_w - 1)/screen_w 
        
    x = int(round((camera.w - 1) * 0.5 * (1 + ndc_x)))
    y = int(round((camera.h - 1) * 0.5 * (1 - ndc_y)))

    return (x, y)

def W2S(camera, W):

    """World-to-screen projection""" 

    cW = W2C(camera, W)  
    return C2S(camera, cW)

def coordinate_on_screen(camera, x):

    arr_shape = (camera.w, camera.h)
    W_1, H_1 = np.array(arr_shape) - np.array([1, 1])

    x_check = x[0] >= 0 and x[0] <= W_1
    y_check = x[1] >= 0 and x[1] <= H_1
    return x_check and y_check

