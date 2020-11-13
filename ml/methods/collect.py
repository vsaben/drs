"""
    Description: Prepares model detection output for visualisation
    Function: 
        Recollect target measurements
              [posx, posy, posz, 
               dimx, dimy, dimz, 
               rotx, roty, rotx, 
               cls] AND
        RPN ROI: normalised [xmin, ymin, xmax, ymax]               
""" 

import tensorflow as tf

from methods._data.reformat import quart_to_euler

class CollectDetection:

    """Collects all detection annotations for a single image"""

    def __init__(self, camera, outs, cfg):

        self.bbox = outs[..., :4]

        self.sco = outs[..., 4]
        self.cls = outs[..., 5]

        # Camera co-ordinates

        self.pos = self.calculate_pos(camera, outs)
        self.dim = self.calculate_dim(outs, cfg)
        self.rot = self.calculate_rot(camera, outs)

    @staticmethod
    def calculate_pos(camera, outs):

        """Calculate relative target position <X, Y, Z> from camera

        :param camera: camera property dictionary
        :param outs: model output per image [ndetect, features]
                     features = [...(6)... + center (2) + depth (1) ...(7)...]

            Let SX, SY = screen x, y
                  X, Y = camera x, y
                
                0.5(1 + ndcx) = SX   |   0.5(1 - ndcy) = SY              
                ndcx = 2SX - 1       |   ndcy = 1- 2SY

                ndcx = fx*(X/Z)      |   ndcy = fy*(Y/Z)
                X = (Z/fx)(2SX - 1)  |   Y = (Z/fy)(1 - 2SY)

                                  
        :result: [ndetect, [posx, posy, posz]]
        """

        SX = outs[..., 6]
        SY = outs[..., 7]
        Z = outs[..., 8]

        X = Z / camera.fx * (2*SX - 1)
        Y = Z / camera.fy * (1 - 2*SY) 

        return tf.concat([X, Y, Z], axis=-1)

    @staticmethod
    def calculate_dim(outs, cfg):

        """Recreates absolute dimensions from dimension anchors (if used), 
        else unaltered predicted dimensions are returned
        
        :param outs: [ndetect, features]
                     features = [...(9)... + dims (3) + ...(4)...]
        :param cfg: model configuration file

        :result: [ndetect, [dimx, dimy, dimz]]
        """

        rdims = outs[..., 9:12]
       
        if not cfg.USE_DIM_ANCHOR:
            return rdims

        med_dims = tf.constant(cfg.MEDIAN_DIMS, tf.float32)
        dims = rdims + med_dims 
        
        return dims

    @staticmethod
    def calculate_rot(outs):

        """Performs quarternion to euler conversion. 
        See 'quart_to_euler' in reformat.py for details.
        
        :param outs: [ndetect, features]
                     features = [...(12)... + quart (4)]
        
        :result: [ndetect, [pitch, roll, yaw]]
        """

        quart = outs[..., -4:]
        euler = tf.map_fn(quart_to_euler, quart, tf.float32)
        return euler

def collect_detections(camera, outs, cfg):

    outs = outs.unbatch()
    dets = [CollectDetection(camera, out, cfg) for out in outs]
    comb = [tf.concat([det.bbox, det.sco, det.cls, det.pos, det.dim, det.rot], axis = -1)
            for det in dets]
    return comb
   

