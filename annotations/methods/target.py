"""
    Description: Extract target annotations. Apply exclusion criteria.
    
    Annotation: (per instance)

        1-3: Position <X, Y, Z> 
             => Target position rel. camera (meters)        
        4-6: Dimension <X (right), Y (forward), Z (up)> 
             => Model dimensions (from center in meters) 
        7-9: Rotation <X (pitch), Y (roll), Z (yaw)> 
             => Relative object rotation. Adapt yaw relative to camera (radians)
        10:  Damage status
             => GTA-detected damage status. Potentially visibly damaged.

    Exclusion criteria:

        a. Disallowed vehicle type
        b. Truncation: Number of bbox vertical edges on screen (incl. center).
                       Points considered: 8 (bbox) + 1 (center)
                       Final score is divided by 2 as vertical edges occur on the 
                       same axis (both are on screen or off screen)
                       Range: (not truncated) 0 - 4.5 (fully truncated)
                       Threshold: <= 2.5
                       Rational: More than half of target is on screen, incl. center
        c. Occlusion: Number of visible target keypoints. Approximated using 
                      stencil image correspondence. Proportion of non-visible
                      keypoints.
                      Range: (not occluded) 0 - 1 (fully occluded)
                      Threshold: <= 2/3
                      Rational: Corresponds well to visible target instances 
                                through trial-and-error. Ensure ssufficient 
                                visibility.

        d. Minimum size: Wide 2D bbox size encapsulating 3D bbox co-ordinates
                         Indication of visible target size
                         Threshold: 25 pixels**2 = 625
                         Rational: Ensures small vehicles that are too small to detect
                                   are excluded.
                         
    Additional features:
        
        i. Vehicle class

    Visualisation: (generated from annotation)
        
        1. 3D BB 
        2. 2D BB
        3. center

    Function:
    - PART A: Target (class)
    - PART B: Generate all image targets
"""

import numpy as np
import enum

from methods import utils, detect

# PART A: Target =====================================================================

class Target(object):
    
    DISALLOWED_VEHICLE_TYPES = {"Boats", "Helicopters", "Military", "Planes", "Trains"}

    class StenCode(enum.Enum):
        person = 1
        car = 2
        bush = 3
        green = 4
        created_car = 130

    OCCLUSION_BONES = {'door_dside_f', 'door_dside_r', 'door_pside_f', 'door_pside_r', 
                       'wheel_lf', 'wheel_rf', 'wheel_lr', 'wheel_rr', 
                       'bumper_f', 'bumper_r', 
                       'bonnet', 'boot' 
                       'headlight_l', 'headlight_r', 'taillight_l', 'taillight_r'}


    def __init__(self, camera, target, ste_arr):        
        
        # [EC]: Vehicle class

        self.vehicleclass = target["identifiers"]["vehicleclass"]
        if self.vehicleclass in Target.DISALLOWED_VEHICLE_TYPES: 
            raise TypeError('EC: Disallowed vehicle type')
        
        # [1-3]: Position <X, Y, Z> 
        #        Target position relative to the camera. Recentered to ensure equal
        #        equal half-dimensions from center (in meters).  
        
        position = target["position"]
        centerW = utils.get_balanced_center(position)
        self.pos = utils.W2C(camera, centerW)
        self.sW = utils.C2S(camera, self.pos)

        # [EC]: Truncation   

        bbox = list(target['bbox'].values())
        is_multi = isinstance(bbox[0], dict) 
        points = [v['Item2'] for v in bbox] if is_multi else bbox + [self.sW] 
        self.truncation = Target.GetTruncation(camera, points)
        if self.truncation > 2.5: raise TypeError('EC: Truncation')

        # [EC]: Occlusion 
        
        bones = target["bones"]
        self.occluded = Target.GetOcclusionLevel(camera, bones, self.sW, ste_arr)
        if self.occluded > 2/3: raise TypeError('EC: Occlusion')
        
        # [4-6]: Dimension <X, Y, Z>                 
        
        self.dims = utils.get_half_dims(position)

        # [7-9]: Rotation (yaw rel. to camera)

        rotation = np.radians(position["rotation"], dtype=np.float64)
        rotation[2] = utils.get_rel_yaw(camera.rot[2], rotation[2], 
                                        is_pos_axis = (camera.R[2, 1] >= 0), 
                                        is_rel_cam=True)
        self.rot = rotation 
  
        # [10]: Potential damage [SM]

        self.damage = (target["damage"] is not None)
    
        # [V]: 3D, wide-2D BB, center  
        
        all_3d = detect.get_all_3d(camera, self.pos, self.dims, self.rot) 
        
        self.bbox3d = detect.get_3d(camera, all_3d)        
        self.bbox2d = detect.get_2d(camera, all_3d)

        # [EC]: Minimum pixel area

        self.pixelarea = Target.GetPixelArea(self.bbox2d)
        if self.pixelarea < 25**2: raise TypeError("EC: Minimum pixel area")

    # Exclusion methods

    @staticmethod
    def GetTruncation(camera, points): 

        """Get target truncation factor (refer description)"""

        points = [p for p in points if utils.coordinate_on_screen(camera, p)] 
        return 4.5 - len(points) / 2

    @staticmethod
    def GetOcclusionLevel(camera, bones, sW, ste_arr):

        """Get target truncation factor (refer description)"""

        ste_h = ste_arr.shape[0]
        f_asp = ste_h/camera.h

        bones = set(tuple(v['Item2']) for k, v in bones.items() 
                    if k in Target.OCCLUSION_BONES and not v['Item2'] == [-1, -1])
        if tuple(sW) not in bones: bones.add(tuple(sW))

        nbones = len(bones)
        if nbones == 0: return 1
        
        nbones_veh = sum([Target.ste_check(ste_arr, b, f_asp) for b in list(bones)])
        return 1 - nbones_veh/nbones 

    @staticmethod
    def ste_check(ste_arr, b, f_asp):      
        
        """Checks whether a pixel in the full resolution colour image corresponds to 
        a vehicle. A 2x2 grid in a lower resolution stencil map is compared to the 
        factored-down pixel position. Pixels relating to vehicles have a 2 or 130 code.  
        """

        float_ste_b = f_asp * np.array(b)                 
        
        shapey, shapex = ste_arr.shape
       
        regx = [max(np.int(np.floor(float_ste_b[0])), 0), 
                min(np.int(np.ceil(float_ste_b[0])), shapex - 1)]
        regy = [max(np.int(np.floor(float_ste_b[1])), 0), 
                min(np.int(np.ceil(float_ste_b[1])), shapey - 1)]

        regx[1] = None if regx[1] == (shapex - 1) else regx[1] + 1  
        regy[1] = None if regy[1] == (shapey - 1) else regy[1] + 1

        ste_reg = ste_arr.T[regx[0]:regx[1], regy[0]:regy[1]]
        ste_che = ste_reg.any()
            #(Target.StenCode.car.value in ste_reg) or (Target.StenCode.created_car.value in ste_reg))       
        return ste_che
     
    @staticmethod
    def GetPixelArea(bb2d):
        xmin, ymin, xmax, ymax = bb2d
        return (xmax - xmin) * (ymax - ymin)

# PART C: Generate all targets ======================================================

def generate_target(camera, target, ste_arr):
    try:
        return Target(camera, target, ste_arr)
    except TypeError:
        return None

def generate_targets(camera, raw_targets, ste_arr):
    targets = []
    for x in raw_targets:
        target = generate_target(camera, x, ste_arr)
        if target is not None: targets.append(target)
    return targets



