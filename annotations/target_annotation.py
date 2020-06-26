# Description: Extract annotation file
# File format: Target
# ---------------------------------------------------------------------------
#   1: Position X  => Target position rel. camera (meters)
#   2: Position Y
#   3: Position Z
#   4: Dimension X => Model dimensions (from center in meters) 
#   5: Dimension Y
#   6: Dimension Z
#   7: Rotation [pitch/alpha] => Model rotation rel. camera (radians)
#   8: Rotation [roll/beta]
#   9: Rotation [yaw/gamma]
# ---------------------------------------------------------------------------
#   10: Truncation => Number of bbox vertical edges on screen (incl. center) 
#   11: Occlusion => 0 [None] --- 1 [< .5] ---- 2 [>= .5]
#   12: Potential Damage => Bool, target is visibly damaged
#   13: Vehicle class
# ---------------------------------------------------------------------------

class Target(object):
    
    def __init__(self, camera, target):

        # [1 - 3]: Position rel. to camera

        C = np.array(camera["C"]["Values"]).reshape((4, 4)).T
        position = target["position"] 
        self.pos = Target.GetPos(C, position)
                 
        # [4 - 6]: Half-dimensions
        
        bbox_max = np.array(position["bbox_max"])
        bbox_min = np.array(position["bbox_min"])
        self.dims = Target.GetDimensions(bbox_min, bbox_max)

        # [7 - 9]: Rotation rel. to camera

        self.rot = np.radians(np.array(position["rotation"]))

        # [10]: Truncation

        center = Target.GetCenter(position)
        bbox = [Target.W2S(C, w) for w in Target.GetCorners(position, center)]     
        bbox.append(Target.C2S(self.pos))       
        self.truncation = Target.GetTruncation(bbox)

        # [11]: Occlusion [M]

        self.occluded = False     

        # [12]: Potential damage [SM]

        self.damage = (target["damage"] is not None)

        # [13] Vehicle class

        self.vehicleclass = target["identifiers"]["vehicleclass"]

    # A [1 - 3]: Get re-centered target position in camera co-ordinates (in meters) 
     
    @staticmethod
    def GetCenter(position):
        W = np.array(position["location"])

        f, r, u = Target.GetDirectionalVectors(position)
        bbox_min = np.array(position["bbox_min"])
        bbox_max = np.array(position["bbox_max"])
        
        x_off, y_off, z_off = bbox_min + bbox_max
        return W + r * x_off + f * y_off + u * z_off

    @staticmethod
    def GetDirectionalVectors(position):
        f = np.array(position["forward_vec"])
        r = np.array(position["right_vec"])
        u = np.array(position["up_vec"])
        return f, r, u

    @staticmethod
    def GetPos(C, position):
        W = Target.GetCenter(position)
        return Target.W2C(C, W)

    @staticmethod
    def W2C(C, W):
        return C[1:, :3] @ W + C[1:, 3]

    # B [4 - 6]: Get half-dimensions off target center (in meters)

    @staticmethod
    def GetDimensions(bbox_min, bbox_max):
        return (bbox_max - bbox_min) / 2

    # C [7 - 9]: Target rotation is extracted directly from GTA (in radius)

    # D [10]: Get the number of vertical edges truncated (incl. center)

    @staticmethod
    def GetTruncation(bbox): 
        bbox_oob = [x for x in bbox if not any([x[0] < 1, x[0] > 1920, x[1] < 1, x[1] > 1080])]
        return len(bbox_oob) / 2

    @staticmethod
    def GetCorners(position, center = np.array([0, 0, 0])):

        f, r, u = Target.GetDirectionalVectors(position)
 
        gmin = np.array(position["bbox_min"])
        gmax = np.array(position["bbox_max"])
 
        corners = np.vstack((gmin[0] * r + gmax[1] * f + gmax[2] * u,  # FTL
                             gmax[0] * r + gmax[1] * f + gmax[2] * u,  # FTR
                             gmin[0] * r + gmax[1] * f + gmin[2] * u,  # FBL
                             gmax[0] * r + gmax[1] * f + gmin[2] * u,  # FBR
                             gmin[0] * r + gmin[1] * f + gmax[2] * u,  # BTL
                             gmax[0] * r + gmin[1] * f + gmax[2] * u,  # BTR
                             gmin[0] * r + gmin[1] * f + gmin[2] * u,  # BBL
                             gmax[0] * r + gmin[1] * f + gmin[2] * u)) # BBR

        corners += center
        return corners

    @staticmethod
    def W2S(C, W):
        cW = C[1:, :3] @ W + C[1:, 3] 
        return Target.C2S(cW)
    
    @staticmethod
    def C2S(cW):

        X, Y, Z = cW

        invW = 1 / Z        
        X *= invW
        Y *= invW
            
        return [int(round(1920 * 0.5 *(1 + X))), int(round(1080 * 0.5 * (1 - Y)))]

    # E: [11] Target is automatically assumed to be not occluded (investigate stencil) 
    # D: [12] Extract damage status from GTA V (check visibility) 
    # F: [13] Vehicle class extracted directly from GTA V

