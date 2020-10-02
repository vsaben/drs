"""
    Description: Training and testing control parameters
    Configuration class:
    - PART A: Name (configuration identifier)
    - PART B: General (paths and auxillary settings)
    - PART C: Image (image resize dimensions and channels)
    - PART D: Yolo (chosen model parameters)
    - PART E: Train
"""

import os
import json
import tensorflow as tf

# Adaptation of Mask RCNN repo config file: https://github.com/matterport/Mask_RCNN
# Glossary:
#   - PI: Performance Improvement > Area for potential PI

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and overrides properties
    where needed.
    """

    NAME = None

    """A: Image ================================================================="""
    
    # Target, square, resized colour image dimensions. Must be divisible by 32 in 
    #    Yolo backbone. Colour images contain 3 channels. [PI]
    
    CHANNELS    = 3
    IMAGE_SIZE  = 416
    
    """B: Training\Inference ===================================================="""

    BATCH_SIZE  = 2
   
    BUFFER_SIZE = 1000

    EPOCHS      = 2

    # Specifies whether an exploratory data analysis is conducted. If True: 
    #   MAX_GT_INSTANCES = 2 * maximum boxes observed in the training data
    #   Yolo's default ANCHORS_6 and ANCHORS_9 are updated. A results 
    #   exploratory.txt file is written to the data directory. Only conducted upon 
    #   a model's initial creation.
 
    EXPLORE_DATA = False

    # Applies a horizontal flip augmentation to each image. Doubles the effective 
    #    dataset size. 

    ISAUGMENT   = True

    # "The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes weights to 
    #   explode. Likely due to differences in optimizer implementation". Reduced 
    #   upon plateau using the associated keras callback. 

    LR_INIT     = 1e-3
    LR_END      = 1e-6

    # Train or freeze batch normalization layers. Poor training implications 
    #   where batch size is small. 

    TRAIN_BN    = False
    
    # Loss components and weights.

    LOSSES = {
        'rpn_loss': 1., 
        'pose_loss': 1. 
        #'mask_loss': 1.
        }

    RPN_TYPE_LOSSES = {
        'rpn_xy_loss':     1.,
        'rpn_wh_loss':     1., 
        'rpn_obj_loss':    1.,
        'rpn_class_loss':  1., 
        'rpn_objcf_loss':  0., 
        'rpn_nobjcf_loss': 0.
        }

    RPN_LVL_LOSSES = {
        'rpn_lvl_1_loss':   1., 
        'rpn_lvl_2_loss':   1., 
        'rpn_lvl_3_loss':   1.
        }

    POSE_SUBLOSSES = {
        'pose_center_loss':1.,
        'pose_depth_loss': 1.,
        'pose_dim_loss':   1., 
        'pose_quart_loss': 1.
        }

    #OD_SUBLOSSES = {
    #   'od_bbox_loss':  1., 
    #   'od_class_loss': 1.
    #    }

    # L2 regularization. Applied after model build.
        
    WEIGHT_DECAY = 0.0001

    # Metrics stored in addition to loss components (see above). RPN losses are 
    #   recorded per output level.

    METRICS = {}


    """C: Backbone =============================================================="""

    # Backbone network architecture. Supported architectures include "yolov3", 
    #    "yolov3t", "yolov4" and "yolov4". Architecture-specific default properties
    #    are stored in the Yolo class.
    
    BACKBONE = 'yolov3'

    # Percentage of Darknet convolutional layers that are trainable (from head to back). 
    #   This prevents unnecessary, computationally-expensive coarse feature relearning 
    #   possessed through transfer learning.

    PER_DARKNET_UNFROZEN = 0.2

    # Non-max suppression threshold to filter RPN proposals. Increase during training 
    #   to generate more valid proposals (if USE_RPN_ROI = True).

    RPN_IOU_THRESHOLD = 0.5

    RPN_SCORE_THRESHOLD = 0.5

    # Use RPN-generated ROIs OR ground-truth ROI during training. 
    #   False: Allow the head to train with perfect information 
    #   True: Permit head-to-rpn performance compensation
   
    USE_RPN_ROI = False

    # Maximum number of ground truth detections to include in a single image during training. 
    #   Set above the maximum number of detections observed in training images 
    #   (with margin). Calculated in exploratory data analysis (see _data_anchors.py).
    #   Permits batching through padding. 

    MAX_GT_INSTANCES = 40     
    
    # Number of object categorisations. Overwritten during name file parsing.

    NUM_CLASSES = 2

    """D: Head ================================================================="""

    # Feature pyramid feature map depth.

    TOP_DOWN_PYRAMID_SIZE = 256

    # ROI pooling size    

    POOL_SIZE = 7
    MASK_POOL_SIZE = 14



    FC_LAYERS_SIZE = 1024

    # Defines dimension measurements relative to the training set's median observed 
    #   vehicle dimensions. Recollected in the final layer. Set once per model. 

    USE_DIM_ANCHORS = False

    USE_DEFAULT_ANCHORS = True

    # Add a bbox and class refinement head module / Separate od from pose

    ADD_POSE_OD = False

    

    """E: Detection ============================================================"""
    
    # Maximum number of possible detections in a single image. See MAX_GT_INSTANCES.
    
    DETECTION_MAX_INSTANCES = 50

    DETECTION_SCORE_THRESHOLD = 0.7

    DETECTION_IOU_THRESHOLD = 0.7


    



    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    # STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    # VALIDATION_STEPS = 50

    # Size of the fully-connected layers in the classification graph
    # FPN_CLASSIF_FC_LAYERS_SIZE = 1024


    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    #USE_MINI_MASK = True
    #MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask


    # Image mean (RGB)
    # MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    # TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier/mask heads
    # Note: YOLO trains negative and positive ROIs simultaneouely
    #ROI_POSITIVE_RATIO = 0.33


    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    #MASK_SHAPE = [28, 28]

    # Bounding box refinement standard deviation for RPN and final detections.
    #RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    #BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])



    # Gradient norm clipping
    #GRADIENT_CLIP_NORM = 5.0

    def __init__(self):
        pass

    def save(self, ckpt_dir):
        
        """Save configuration attributes (excl. YOLO) to .json file
        
        :param ckpt_dir: general model saving directory
        :param name: model/test name
        
        :result: configuration attributes (excl. YOLO) stored in config.json
        """

        attrs = [attr for attr in dir(self) if is_attr_of_interest(self, attr)] 
        attr_dict = getattrs(self, attrs)

        file_name = os.path.join(ckpt_dir, self.NAME, "config.json")
        with open(file_name, 'w') as f:
            json.dump(attr_dict, f, sort_keys=True, indent=4)       

    def restore(self, model_dir):
        pass

    def display(self):
        
        """Display configuration values"""
        
        print("\nConfigurations:")
        for attr in dir(self):
            if is_attr_of_interest(self, attr): 
                if attr == 'YOLO':
                    print("\n")
                    self.YOLO.display()
                    print("\n")
                else:
                    print("{:30} {}".format(attr, getattr(self, attr)))
        print("\n")


class Yolo(object):               
    MASKS       = tf.constant([[6, 7, 8], [3, 4, 5], [0, 1, 2]], tf.int32)              # Sequential layers (3) are responsible for predicting masks of decreasing size 
    DIM_ANCHOR  = tf.constant([0.9889339804649353, 2.4384219646453857, 0.7416530251502991], tf.float32)

    ANCHORS_9 = tf.constant([(1, 8),(2, 8),(2, 2),(6, 26),(7, 10),(11, 16),(18, 20),(24, 44),(68, 134)], tf.float32) / 416
    ANCHORS_6 = tf.constant([(2, 7),(8, 13),(17, 19),(19, 39),(34, 62),(91, 156)], tf.float32) / 416

    def __init__(self, cfg, ckpt_dir):

        """
        :param backbone: backbone architecture (string)
        :param isdefault: default or calculated anchors
        """

        model = self.get_base_model(cfg.BACKBONE)

        Yolo.NAME = model.NAME
        
        istiny = (Yolo.NAME[-1] == "T")        
        if not cfg.USE_DEFAULT_ANCHORS: 
            model.ANCHORS = Yolo.ANCHORS_6 if istiny else Yolo.ANCHORS_9

        model_dir = os.path.join(ckpt_dir, cfg.NAME)
        if os.path.isdir(model_dir):
            Yolo.CKPT = os.path.join(model_dir, "model.ckpt")
        else:
            Yolo.CKPT = os.path.join(ckpt_dir, "{:s}-416.ckpt".format(Yolo.NAME.lower()))
        
        Yolo.ANCHORS = model.ANCHORS           
        if istiny: Yolo.MASKS = Yolo.MASKS[1:]         
        

    def get_base_model(self, backbone):

        """Get the base Yolo model class based on the backbone selected 
        
        :param backbone: yolov3, yolov3t, yolov4, yolov4t  

        :result: chosen backbone' configuration settings        
        """

        if backbone == "yolov3": return Yolo.V3
        if backbone == "yolov3t": return Yolo.V3T
        if backbone == "yolov4": return Yolo.V4
        if backbone == "yolov4t": return Yolo.V4T

    class V3:
        NAME      = "YoloV3"
        ANCHORS   = tf.constant([(10,13),(16,30),(33,23),(30,61),(62,45),(59,119),(116,90),(156,198),(373,326)],tf.float32) / 416 # Anchors specified relative to W = H as a proportion [0 - 1]             
        STRIDES   = [8, 16, 32]
        XYSCALE   = [1.2, 1.1, 1.05]
            
    class V3T:
        NAME      = "YoloV3T"
        ANCHORS   = tf.constant([(10,14),(23,27),(37,58),(81,82),(135,169),(344,319)], tf.float32) / 416     
        STRIDES   = [16, 32]
        XYSCALE   = [1.05, 1.05]

    class V4:
        NAME      = "YoloV4"
        ANCHORS   = tf.constant([(12,16),(19,36),(40,28),(36,75),(76,55),(72,146),(142,110),(192,243),(459,401)],tf.float32) / 608

    class V4T:
        NAME      = "YoloV4T"
        ANCHORS   = tf.constant([(10,14),(23,27),(37,58),(81,82),(135,169),(344,319)], tf.float32) / 416

    
def setattrs(_self, **kwargs):
    for k, v in kwargs.items():
        setattr(_self, k, v)

def getattrs(_self, attrs):
    return {k: getattr(_self, k) for k in attrs}

def is_attr_of_interest(_self, attr):
    return not attr.startswith("__") and not callable(getattr(_self, attr)) and attr != 'YOLO'
