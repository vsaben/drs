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
import numpy as np
import tensorflow as tf

# Adaptation of Mask RCNN repo config file: https://github.com/matterport/Mask_RCNN
# Glossary:
#   - PI: Performance Improvement > Area for potential PI

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and overrides properties
    where needed.
    """

    # Model name. All logs, checkpoints and other model-specific outputs are stored 
    #   in a created model directory of the same name.

    NAME = None

    BEST_VAL_LOSS = None

    TRAIN_SESSIONS = 0

    """A: Image ================================================================="""
    
    # Target, square, resized colour image dimensions. Must be divisible by 32 in 
    #    Yolo backbone. Colour images contain 3 channels. Increasing target image 
    #    dimensions favours accuracy (i.e. improves feature resolution) in the 
    #    speed-accuracy trade-off. [PI]
    
    CHANNELS = 3
    IMAGE_SIZE = 416
    
    """B: Training\Inference ===================================================="""

    # Batch size balances error gradient accuracy against convergence speed. While 
    #   larger batches enhance error gradient estimations, convergence is slower 
    #   (and vice versa). Limited here by GPU memory constraints.   

    BATCH_SIZE = 32
   
    # Shuffle buffer size. Affects the randomness in order to which instances are 
    #   selected and batched. Let b = buffer size. An instance is drawn from a 
    #   pseudo-randomly selected subset of b instances. The next item then replaces 
    #   the selected instance. 3 scenarios persist:
    #   i. b < len(data) --> data not fully randomised / smaller memory overhead
    #   ii. b >= len(data) --> data fully shuffled / large memory overhead
    #   iii. b = 1 --> no shuffling
    #   A "sufficiently large" buffer size is selected to maximise data order 
    #   randomness whilst considering memory constraints. 

    BUFFER_SIZE = 1000

    # Number of training iterations through the full dataset. Early stopping 
    #   permitted via the early stopping keras callback. Total epochs stored for 
    #   to combine tensorboard summaries via a step parameter.   

    EPOCHS = 2

    TOTAL_EPOCHS = 0

    # Specifies if an exploratory data analysis is conducted. If true, the analyis 
    #   is performed upon a model's initial creation. The following attributes are 
    #   updated in the configuration file:
    #   > MAX_GT_INSTANCES = 2 * maximum boxes observed in the training data
    #   > ANCHORS
    #   > DIM_ANCHOR 
    #   More detailed data characteristics are written to 'exploratory.txt' in the 
    #   data directory.  
 
    EXPLORE_DATA = True

    # Applies a horizontal flip augmentation to each image. Doubles the effective 
    #    dataset size. 

    ISAUGMENT = True

    # Gradient error scaling factor. "The Mask RCNN paper uses lr=0.02, but on 
    #   TensorFlow it causes weights to explode. Likely due to differences in 
    #   optimizer implementation". Reduced upon plateau using the associated keras 
    #   callback.  

    LR_INIT = 1e-3
    LR_END = 1e-6

    # Train or freeze batch normalization layers. Poor training implications 
    #   where batch size is small. 

    TRAIN_BN    = True
    
    # Loss components and weights. Component weighting focuses model performance. 
    #   All components are provided as metrics. RPN metrics account for all 
    #   level-type pairs. The number of RPN level losses measured depends on the
    #   backbone's number of output layers.

    LOSSES = {
        'rpn_loss':        1., 
        'pose_loss':       1.
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
        'rpn_lvl_0_loss':   1., 
        'rpn_lvl_1_loss':   1., 
        'rpn_lvl_2_loss':   1.
        }

    POSE_SUBLOSSES = {
        'pose_center_loss':1.,
        'pose_depth_loss': 1.,
        'pose_dim_loss':   1., 
        'pose_quart_loss': 1.
        }

    # L2 regularisation factor. Applied after model build.
        
    WEIGHT_DECAY = 0.0001

    # Metrics stored in addition to loss components (see above).

    METRICS = {}

    # Class weighting accounts for the distribution imbalance of damaged to undamaged 
    #  vehicle instances in the data. Damaged vehicles are less frequent. Class losses 
    #  are weighted to ensure classification decisions rely on features, as opposed to 
    #  class modality. Focal loss is chosen. This is empirically shown to improve model
    #  performance on autonomous vehicle datasets: "Resolving Class Imbalance in Object 
    #  Detection with Weighted Cross Entropy Losses". w = (1 - P(i))^a where 
    #   a = hyperparameter
    #   P(i) = class propensity
    #  Smaller classes are weighted higher, thereby focusing model performance on their 
    #  accurate classification. Updated in exploratory data analysis. Additional 
    #  investigation is left for future research. [PI] 

    DAMAGED_RATIO = 0.2115
    
    FOCAL_LOSS_ALPHA = 1

    """C: Backbone =============================================================="""

    ANCHORS = None

    DEFAULT_YOLO_WEIGHT_PATH = None

    MASKS = None

    # Backbone network architecture. Supported architectures include "YoloV3", 
    #    "YoloV3T", "YoloV4" and "YoloV4T". Architecture-specific default properties
    #    are stored in the Yolo class.
    
    BACKBONE = 'YoloV3'

    # Percentage of Darknet convolutional layers that are trainable (from head to back). 
    #   This prevents unnecessary, computationally-expensive coarse feature relearning 
    #   possessed through transfer learning.

    PER_DARKNET_UNFROZEN = 0.2

    # Non-max suppression thresholds to filter RPN proposals. Increase during training 
    #   to generate more valid proposals (if USE_RPN_ROI = True).

    RPN_IOU_THRESHOLD = 0.5
    RPN_SCORE_THRESHOLD = 0.5

    # Use RPN-generated ROIs OR ground-truth ROI during training. 
    #   False: Allow the head to train with perfect information 
    #   True: Permit head-to-rpn performance compensation
   
    USE_RPN_ROI = False

    # Maximum number of ground truth detections to include in a single image during 
    #   training. Set above the maximum number of detections observed in training 
    #   images (with margin). Calculated in exploratory data analysis 
    #   (see EXPLORE_DATA). Permits batching through padding. 

    MAX_GT_INSTANCES = 40     
    
    # Number of object categorisations. Overwritten during name file parsing.

    NUM_CLASSES = 2

    CLASSES = ['undamaged', 'damaged']

    """D: Head ================================================================="""

    # Feature pyramid feature map depth. Manages model size.

    TOP_DOWN_PYRAMID_SIZE = 256

    # ROI pooling size    

    POOL_SIZE = 7
    #MASK_POOL_SIZE = 14



    FC_LAYERS_SIZE = 1024

    # Defines dimension measurements relative to the training set's median observed 
    #   vehicle dimensions. Recollected in the final layer. Set once per model. 

    USE_DIM_ANCHOR = True
    DIM_ANCHOR = [0.9889339804649353, 2.4384219646453857, 0.7416530251502991]

    USE_DEFAULT_ANCHORS = False

    """E: Optimisation ========================================================="""

    # Quantisation prunes weights by reducing their float precision. Options: 
    #   ['none', 'aware', 'post', 'both']

    QUANTISATION = 'none'

    """F: Detection ============================================================"""
    
    # Maximum number of possible detections in a single image. See MAX_GT_INSTANCES.

    DET_MAX_GT_INSTANCES = 50

    DET_RPN_SCORE_THRESHOLD = 0.7

    DET_RPN_IOU_THRESHOLD = 0.7

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

    def save(self):
        
        """Save configuration attributes (excl. YOLO) to .json file
        
        :param ckpt_dir: general model saving directory
        :param name: model/test name
        
        :result: configuration attributes stored in config.json
        """

        if not os.path.isdir(self.MODEL_DIR): 
            os.mkdir(self.MODEL_DIR)

        file_name = os.path.join(self.MODEL_DIR, "config.json")        
        with open(file_name, 'w') as f:
            json.dump(self.to_dict(), f, sort_keys=True, indent=4)  
            
    def restore(model_dir):
        file_name = os.path.join(model_dir, "config.json")
        with open(file_name, 'r') as f:
            cfg_dict = json.load(f)
        
        cfg = Config()
        
        for k, v in cfg_dict.items():
            setattr(cfg, k, v)
       
        return cfg 

    def display(self, mode):
        
        """Display configuration values"""
        
        mode_cond = lambda attr: attr.startswith('DET_') if mode == 'detection' else True    

        print("\nConfigurations:")
        for attr in dir(self):
            if attr.isupper() and mode_cond(attr): 
                print("{:30} {}".format(attr, getattr(self, attr)))
        print("\n")

    def to_dict(self):
        return {attr: getattr(self, attr) for attr in dir(self) if attr.isupper()}

def UpdateConfigYolo(cfg):               
    
    """
    :param backbone: backbone architecture (string)
    :param isdefault: default or calculated anchors
    """

    yolo = Yolo(cfg)

    setattrs(cfg, ANCHORS = yolo.ANCHORS, 
                  MASKS = yolo.MASKS, 
                  DEFAULT_YOLO_WEIGHT_PATH = yolo.DEFAULT_YOLO_WEIGHT_PATH)          

class Yolo:
    MASKS     = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]             # Sequential layers (3) are responsible for predicting masks of decreasing size 
    
    ANCHORS_9 = [np.array([(1, 8),(2, 8),(2, 2),(6, 26),(7, 10),(11, 16),(18, 20),(24, 44),(68, 134)], np.float32) / 416][0].tolist()
    ANCHORS_6 = [np.array([(2, 7),(8, 13),(17, 19),(19, 39),(34, 62),(91, 156)], np.float32) / 416][0].tolist()

    def __init__(self, cfg, default_ckpt_dir = "./data/weights/"):
        model = self.get_base_model(cfg.BACKBONE)

        istiny = (cfg.BACKBONE[-1] == "T") 
        self.ANCHORS = model.ANCHORS
        
        if not cfg.USE_DEFAULT_ANCHORS: 
            self.ANCHORS = Yolo.ANCHORS_6 if istiny else Yolo.ANCHORS_9

        self.MASKS = Yolo.MASKS[1:] if istiny else Yolo.MASKS 
        self.DEFAULT_YOLO_WEIGHT_PATH = os.path.join(default_ckpt_dir, "{:s}.weights".format(cfg.BACKBONE.lower()))

    def get_base_model(self, backbone):

        """Get the base Yolo model class based on the backbone selected 
        
        :param backbone: yolov3, yolov3t, yolov4, yolov4t  

        :result: chosen backbone' configuration settings        
        """

        if backbone == "YoloV3": return Yolo.V3
        if backbone == "YoloV3T": return Yolo.V3T
        if backbone == "YoloV4": return Yolo.V4
        if backbone == "YoloV4T": return Yolo.V4T

    class V3:
        ANCHORS   = [np.array([(10,13),(16,30),(33,23),(30,61),(62,45),(59,119),(116,90),(156,198),(373,326)], np.float32) / 416][0].tolist() # Anchors specified relative to W = H as a proportion [0 - 1]             
        STRIDES   = [8, 16, 32]
        XYSCALE   = [1.2, 1.1, 1.05]
            
    class V3T:
        ANCHORS   = [np.array([(10,14),(23,27),(37,58),(81,82),(135,169),(344,319)], np.float32) / 416][0].tolist()     
        STRIDES   = [16, 32]
        XYSCALE   = [1.05, 1.05]

    class V4:
        ANCHORS   = [np.array([(12,16),(19,36),(40,28),(36,75),(76,55),(72,146),(142,110),(192,243),(459,401)], np.float32) / 608][0].tolist()

    class V4T:
        ANCHORS   = [np.array([(10,14),(23,27),(37,58),(81,82),(135,169),(344,319)], np.float32) / 416][0].tolist()

    
def setattrs(_self, **kwargs):
    for k, v in kwargs.items():
        setattr(_self, k, v)

class ConfigModel:
    pass


# Add bbox and class refinement to the head graph. 3 configurations are permitted:
#  - "separate": separate graph computed in parallel 
#  - "pose": incorporated into the pose's graph output layer
#  - "none": no head level object detection refinement is performed
# Abandoned owing to the 3D focus

# use rpn roi
# - ideas: train use rpn roi to enable pose detection based on possible inferior proposal
#           regions at test time
# - abandoned: train to learn features based on perfect information.   


