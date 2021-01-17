"""
    Description: Model training control panel
    Function:
    - PART A: Manage flags
    - PART B: Update configuration class
    - PART C: Load data
    - PART D: Configure model
"""

from absl import app, flags, logging
from absl.flags import FLAGS

import os
import numpy as np

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

# tf.debugging.enable_check_numerics(stack_height_limit=60, path_length_limit=100) # ADJUST (Remove when finished)

from config import Config, UpdateConfigYolo, setattrs
from methods._data.exploratory import explore
from methods.data import load_all_ds
from methods.model import DRSYolo 

# A: Manage flags =================================================================================================

flags.DEFINE_string('name',         "default",           'configuration name')

flags.DEFINE_string('data_path',    "./data/",           'path to training and validation data folders')
flags.DEFINE_string('model_dir',    "./models/",         'storage directory for all models')
flags.DEFINE_string('class_path',   "./data/drs.names",  'path to classes file')

flags.DEFINE_string('backbone',     Config.BACKBONE,     'YoloV3, YoloV3T, YoloV4 or YoloV4T')

flags.DEFINE_integer('image_size',  Config.IMAGE_SIZE,   'image input size')
flags.DEFINE_integer('epochs',      Config.EPOCHS,       'number of training epochs')                                            
flags.DEFINE_integer('batch_size',  Config.BATCH_SIZE,   'training batch size')                      

def main(_argv):  

    # B: Update configuration class =======================================================================

    model_dir = os.path.join(FLAGS.model_dir, FLAGS.name)       
    log_dir = os.path.join(model_dir, "logs")

    isnew = not os.path.isdir(model_dir)

    if not isnew:
        cfg = Config.restore(model_dir)       
        setattrs(cfg, EPOCHS = FLAGS.epochs, 
                      BATCH_SIZE = FLAGS.batch_size) 
    else:

        if FLAGS.image_size % 32 != 0: raise Exception("Image size must be perfectly divisible by 32 (Yolo)")
        
        os.mkdir(model_dir)

        with open(FLAGS.class_path) as f:
            CLS = [cls.strip() for cls in f.readlines()]
            NUM_CLS = len(CLS)

        cfg = Config()
        setattrs(cfg, NAME = FLAGS.name, 
                      IMAGE_SIZE = FLAGS.image_size, 
                      EPOCHS = FLAGS.epochs,
                      BATCH_SIZE = FLAGS.batch_size,
                      CLASSES = CLS,
                      NUM_CLASSES = NUM_CLS,
                      BEST_VAL_LOSS = np.inf)

        UpdateConfigYolo(cfg)

        isexplore = os.path.isfile(os.path.join(FLAGS.data_path, 'exploratory.txt'))
        if cfg.EXPLORE_DATA and not isexplore:        
            explore(FLAGS.data_path, cfg.IMAGE_SIZE, nclusters = 14, cfg = cfg)

    setattrs(cfg, MODEL_DIR = model_dir, 
                  LOG_DIR = log_dir)

    cfg.display('training')
  
    # C: Load data ========================================================================================
           
    with tf.device('/CPU:0'):
        train_ds, val_ds, infer_ds = load_all_ds(FLAGS.data_path, cfg = cfg)

    # D: Configure and fit model ===========================================================================

    cfg_mod = cfg.to_dict()

    model = DRSYolo(mode="training", cfg=cfg_mod, infer_ds=infer_ds) if isnew else ( 
            DRSYolo.restore(cfg=cfg_mod, infer_ds=infer_ds))
    
    model.build()
    model.summary()
    
    history = model.fit(train_ds, val_ds)

    model.save(name='model', full=False, schematic=isnew)
    #model.optimise()
    
    val_losses = history.history['val_loss'] + [cfg.BEST_VAL_LOSS]
    setattrs(cfg, TOTAL_EPOCHS = cfg.TOTAL_EPOCHS + cfg.EPOCHS, 
                  BEST_VAL_LOSS = np.nanmin(val_losses), 
                  TRAIN_SESSIONS = cfg.TRAIN_SESSIONS + 1)
    cfg.save()

    """
    import tensorflow as tf
    import os
    import numpy as np
    from config import Config, UpdateConfigYolo, setattrs
    from methods.data import load_ds, load_inference_ds
    from methods.model import DRSYolo 

    model_dir = "./models/default"
    isnew = not os.path.isdir(model_dir)
    
    if isnew:

        os.mkdir(model_dir)

        with open("./data/drs.names") as f:
            CLS = [cls.strip() for cls in f.readlines()]
            NUM_CLS = len(CLS)

        cfg = Config()
        setattrs(cfg, NAME = "default",  
                      EPOCHS = 2,
                      BATCH_SIZE = 4,
                      BEST_VAL_LOSS = np.inf)

        UpdateConfigYolo(cfg)

    else:

        cfg = Config.restore(model_dir)    
        setattrs(cfg, EPOCHS = 2, 
                      BATCH_SIZE = 4, 
                      MODEL_DIR = "./models/default", 
                      LOG_DIR = "./models/default/logs") 
    
    cfg.display('training')

    with tf.device('/CPU:0'):
        train_ds = load_ds("./data/", istrain = True, cfg=cfg)
        val_ds = load_ds("./data/", istrain = False, cfg=cfg)

        regex = os.path.join("./data/", "inference") + "/*.tfrecord"
        sample_ds = load_inference_ds(regex, cfg = cfg)

    #x = list(val_ds.take(3))[0]    
    #print(x)

    cfg_mod = cfg.to_dict()
    model = DRSYolo(mode="training", cfg=cfg_mod, sample_ds=sample_ds)        
    model.build()

    #model.summary()

    history = model.fit(train_ds, val_ds)
    """

    """OUT: MODEL.FIT

    tf.config.experimental_run_functions_eagerly(True)    
    history = model.fit(train_ds, val_ds)
    """  

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


# ADJUST: Apply automated class reweighting
# REFERENCE: flags.DEFINE_integer('weights_num_classes', None, 'specify num class for `weights` file if different, '
#                                 'useful in transfer learning with different number of classes')
