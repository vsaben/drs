# Description: Annotation file and visualisation

# ---------------------------------------------------------------------------
# ANNOTATION FILE 
# ---------------------------------------------------------------------------
#   1: Position X  => Target position rel. camera (meters)
#   2: Position Y
#   3: Position Z
#   4: Dimension X => Model dimensions (from center in meters) 
#   5: Dimension Y
#   6: Dimension Z
#   7: Rotation [pitch/alpha] => Model rotation w/ yaw relative to camera (radians)
#   8: Rotation [roll/beta]
#   9: Rotation [yaw/gamma] 
# ---------------------------------------------------------------------------
#   10: Truncation => Number of bbox vertical edges on screen (incl. center) 
#   11: Occlusion => 0 [None] --- 1 [< .5] ---- 2 [>= .5]
#   12: Potential Damage => Bool, target is visibly damaged
#   13: Vehicle class
#   14: Vehicle pixel area
# ---------------------------------------------------------------------------
# VISUALISATION
# ---------------------------------------------------------------------------
#    a: 3D bbox
#    b: 2D bbox   
#    c: Vehicle center 
# ---------------------------------------------------------------------------
# ADMISSIBLE TARGET 
# ---------------------------------------------------------------------------
#    a: Occlusion 
#    b: Size
    
import numpy as np
from math import sin, cos, pi, tan

from enum import Enum
class StenCode(Enum):
    person = 1
    car = 2
    bush = 3
    green = 4

class Target(object):
    
    def __init__(self, camera, target, ste_arr):
        
        # Annotations -------------------------------------------------------
        
        # [1 - 3]: Position rel. to camera
        
        position = target["position"]
        centerW = Target.GetCenter(position)
        self.pos = Target.W2C(camera, centerW)
        sW = Target.C2S(camera, self.pos)

        # CHECK 1: Maximum occlusion
        
        bones = target["bones"]
        self.occluded = Target.GetOcclusionLevel(camera, bones, sW, ste_arr)

        if self.occluded > 3/4: raise TypeError('Check 1: Target is too occluded')
        
        # CHECK 2: Minimum size

        # [4 - 6]: Half-dimensions
        
        bbox_max = np.array(position["bbox_max"], dtype=np.float64)
        bbox_min = np.array(position["bbox_min"], dtype=np.float64)
        self.dims = Target.GetDimensions(bbox_min, bbox_max)

        # [7 - 9]: Rotation rel. to camera

        rotation = np.radians(position["rotation"], dtype=np.float64)
        rotation[2] = Target.GetRelYaw(camera, rotation[2], True)
        self.rot = rotation 

        # [10]: Truncation

        bbox = list(target['bbox'].values())
        is_multi = isinstance(bbox[0], dict) 
        points = [v['Item2'] for v in bbox] if is_multi else bbox + [sW] 
        self.truncation = Target.GetTruncation(points)

        # [11]: Occlusion (refer check 1)     
        # [12]: Potential damage [SM]

        self.damage = (target["damage"] is not None)

        # [13] Vehicle class

        self.vehicleclass = target["identifiers"]["vehicleclass"]
    
        # Visualisation -----------------------------------------------------
        
        all_bb3d = Target.get_all_bb3d(camera, self.pos, self.dims, self.rot) 
        self.bbox3d = Target.Get3DBB(all_bb3d)        
        self.bbox2d = Target.Get2DBB(all_bb3d)
        self.pixelarea = Target.GetPixelArea(self.bbox2d)
        self.sW = sW

    # 0: Helper functions 
    
    @staticmethod
    def get_directional_vectors(position):
        f = np.array(position["forward_vec"], dtype=np.float64)
        r = np.array(position["right_vec"], dtype=np.float64)
        u = np.array(position["up_vec"], dtype=np.float64)
        return f, r, u
    
    @staticmethod
    def create_rot(euler):

        # Notes: Creates [[right], [forward], [up]] base vectors
        #        Switch to [-[right], -[up], [forward]] for camera 

        x, y, z = euler

        Rx = np.array([[1, 0, 0],
                       [0, cos(x), -sin(x)],
                       [0, sin(x), cos(x)]], dtype=np.float64)
        Ry = np.array([[cos(y), 0, sin(y)],
                       [0, 1, 0],
                       [-sin(y), 0, cos(y)]], dtype=np.float64)
        Rz = np.array([[cos(z), -sin(z), 0],
                       [sin(z), cos(z), 0],
                       [0, 0, 1]], dtype=np.float64)
        
        R = Rz @ Ry @ Rx
        return R
    
    @staticmethod
    def get_corner_dim(dims):     
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

    @staticmethod
    def coordinate_on_screen(x, iscol = True):

        W, H = (1920, 1080) if iscol else (1280, 720)
        x_check = x[0] >= 1 and x[0] <= W
        y_check = x[1] >= 1 and x[1] <= H
        return x_check and y_check

    @staticmethod
    def get_all_bb3d(camera, cW, dims, tar_rot):
        
        tar_cam_rot = np.array([tar_rot[0],
                                tar_rot[1],
                                Target.GetRelYaw(camera, tar_rot[2], False)], dtype=np.float64)   
        R_tar = Target.create_rot(tar_cam_rot)

        corner_dims_dict = Target.get_corner_dim(dims)
        corner_dims = list(corner_dims_dict.values())
        corner_dims = np.vstack(corner_dims).T
        
        corner_cam = (camera.R.T @ R_tar @ corner_dims).T + cW  
        return {k: Target.C2S(camera, c) for k, c in zip(corner_dims_dict.keys(), corner_cam)}                
      
    @staticmethod
    def W2C(camera, W):
        return camera.R.T @ (W - camera.pos)

    @staticmethod
    def C2S(camera, cW):

        ndc_x = camera.fx * cW[0]/cW[2] 
        ndc_y = camera.fy * cW[1]/cW[2]

        return [int(round(camera.w * 0.5 *(1 + ndc_x))), int(round(camera.h * 0.5 * (1 - ndc_y)))]

    @staticmethod
    def W2S(camera, W):
        cW = W2C(camera, W)  
        return Target.C2S(camera, cW)

    # A [1 - 3]: Get re-centered target position in camera co-ordinates (in meters) 

    @staticmethod
    def GetCenter(position):
        W = np.array(position["location"], dtype=np.float64)

        f, r, u = Target.get_directional_vectors(position)
        bbox_min = np.array(position["bbox_min"], dtype=np.float64)
        bbox_max = np.array(position["bbox_max"], dtype=np.float64)
        
        x_off, y_off, z_off = bbox_min + bbox_max
        return W + r * x_off + f * y_off + u * z_off

    # B [4 - 6]: Get half-dimensions off target center (in meters)

    @staticmethod
    def GetDimensions(bbox_min, bbox_max):
        return (bbox_max - bbox_min) / 2

    # C [7 - 9]: Calculate relative target yaw (in radians)

    @staticmethod
    def GetRelYaw(camera, tar_yaw, is_tar_to_cam):
        
        dir_yaw = camera.R[2, 1] > 0

        if is_tar_to_cam:            
            adj_yaw = Target.realign_rad(tar_yaw - camera.yaw)
            if not dir_yaw: adj_yaw = Target.realign_yaw(adj_yaw, True)
        else: 
            if not dir_yaw: tar_yaw = Target.realign_yaw(tar_yaw, False)
            adj_yaw = tar_yaw + camera.yaw 
            
        return adj_yaw
     
    @staticmethod
    def realign_yaw(theta, is_tar_to_cam):

        # Target-to-cam: <vertical-bottom> <--> <vertical-top> 

        f = pi if is_tar_to_cam else -pi 
        if theta >= -pi and theta <= 0: return theta + f
        if theta > 0 and theta <= pi: return theta - f

    @staticmethod
    def realign_rad(rad):
        if rad >= -pi and rad <= pi: return rad
        if rad < -pi and rad >= -2*pi: return rad + 2*pi
        if rad > pi and rad <= 2*pi : return rad - 2*pi

    # D [10]: Get the number of vertical edges truncated (incl. center)

    @staticmethod
    def GetTruncation(points):  
        points = [p for p in points if Target.coordinate_on_screen(p)] 
        return 4.5 - len(points) / 2

    # E: [11] Target is automatically assumed to be not occluded (investigate stencil)
    
    OCCLUSION_BONES = {'door_dside_f', 'door_dside_r', 'door_pside_f', 'door_pside_r', 
                       'wheel_lf', 'wheel_rf', 'wheel_lr', 'wheel_rr', 
                       'bumper_f', 'bumper_r', 
                       'bonnet', 'boot' 
                       'headlight_l', 'headlight_r', 'taillight_l', 'taillight_r'
                       }

    @staticmethod
    def GetOcclusionLevel(camera, bones, sW, ste_arr):

        ste_h = ste_arr.shape[0]
        f_asp = ste_h/camera.h

        bones = [v['Item2'] for k, v in bones.items() if k in Target.OCCLUSION_BONES and not v['Item2'] == [-1, -1]]
        if sW not in bones: bones.append(sW)

        nbones = len(bones)
        if nbones == 0: return 1
        
        nbones_veh = sum([Target.ste_check(ste_arr, b, f_asp) for b in bones])
        return 1 - nbones_veh/nbones 

    @staticmethod
    def get_ste_reg(ste_arr, ste_b):

        shape_y, shape_x = ste_arr.shape
        reg_x = [max(ste_b[0] - 1, 0), min(ste_b[0] + 1, shape_x - 1) + 1]
        reg_y = [max(ste_b[1] - 1, 0), min(ste_b[1] + 1, shape_y - 1) + 1]

        for reg, shp in zip([reg_x, reg_y], [shape_x, shape_y]):
            if reg[1] == shp: reg[1] = -1 

        return ste_arr.T[reg_x[0]:reg_x[1], reg_y[0]:reg_y[1]]

    @staticmethod
    def ste_check(ste_arr, b, f_asp):        
        ste_b = f_asp * (np.array(b) - np.array([-1, -1]))  
        ste_b = [int(round(ste_b[0])), int(round(ste_b[1]))]       
        
        ste_reg = Target.get_ste_reg(ste_arr, ste_b)
        return (StenCode.car.value in ste_reg) or (130 in ste_reg)

 
    # D: [12] Extract damage status from GTA V (check visibility) 
    # F: [13] Vehicle class extracted directly from GTA V

    # Visualisation ------------------------------------------------------------------

    @staticmethod
    def Get3DBB(all_bb3d):       
        bb3d_onscreen = {k: v for k, v in all_bb3d.items() if Target.coordinate_on_screen(v)}
        return bb3d_onscreen
              
    @staticmethod
    def Get2DBB(all_bb3d):
        
        bb3d = tuple(all_bb3d.values())
        bb3d = np.vstack(bb3d)

        x_vals = bb3d[:, 0]
        y_vals = bb3d[:, 1]

        xmin = np.min(x_vals)
        xmax = np.max(x_vals)
        ymin = np.min(y_vals)
        ymax = np.max(y_vals)

        if xmin < 1: xmin = 1
        if xmax > 1920: xmax = 1920
        if ymin < 1: ymin = 1
        if ymax > 1080: ymax = 1080

        return [xmin, ymin, xmax, ymax]

    @staticmethod
    def GetPixelArea(bb2d):
        xmin, ymin, xmax, ymax = bb2d
        return (xmax - xmin) * (ymax - ymin)

    # ADMISSIBLE TARGETS -----------------------------------------------------------

def GenerateTarget(camera, target, ste_arr):
    try:
        return Target(camera, target, ste_arr)
    except TypeError:
        return None

def GenerateTargets(camera, raw_targets, ste_arr):
    targets = []
    for x in raw_targets:
        target = GenerateTarget(camera, x, ste_arr)
        if target is not None: targets.append(target)
    return targets



