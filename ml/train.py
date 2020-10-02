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

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

from config import Config, Yolo, setattrs
from methods._data_exploratory import explore
from methods.data import load_ds
from methods.model import DRSYolo 

cfg = Config() 

# A: Manage flags =================================================================================================

flags.DEFINE_string('name',         "default",           'configuration name')

flags.DEFINE_string('data_path',    "./data/",           'path to training and validation data folders')
flags.DEFINE_string('ckpt_dir',     "./checkpoints/",    'path to model directory (contains weights/checkponts)')
flags.DEFINE_string('class_path',   "./data/drs.names",  'path to classes file')

flags.DEFINE_string('backbone',     cfg.BACKBONE,        'yolov3, yolov3t, yolov4 or yolov4t')

flags.DEFINE_integer('image_size',  cfg.IMAGE_SIZE,      'image input size')
flags.DEFINE_integer('epochs',      cfg.EPOCHS,          'number of training epochs')                                            
flags.DEFINE_integer('batch_size',  cfg.BATCH_SIZE,       'training batch size')                      

def main(_argv):  

    # B: Update configuration class =======================================================================

    model_dir = os.path.join(FLAGS.ckpt_dir, FLAGS.name)

    #if os.path.isdir(model_dir):
    #    cfg = cfg.restore(model_dir)
    #    cfg.YOLO = Yolo.restore(model_dir)
    #else:

    if FLAGS.image_size % 32 != 0: raise Exception("Image size must be perfectly divisible by 32 (Yolo)")

    with open(FLAGS.class_path) as f:
        NUM_CLS = sum(1 for line in f)

    setattrs(cfg, NAME = FLAGS.name, 
                  IMAGE_SIZE = FLAGS.image_size, 
                  EPOCHS = FLAGS.epochs,
                  BATCH_SIZE = FLAGS.batch_size,
                  NUM_CLASSES = NUM_CLS)

    cfg.YOLO = Yolo(cfg, FLAGS.ckpt_dir)

    if cfg.EXPLORE_DATA:        
        explore(FLAGS.data_path, FLAGS.data_path, cfg.IMAGE_SIZE, nclusters = 14, cfg = cfg)
        os.mkdir(model_dir)      
        cfg.save(FLAGS.ckpt_dir)

    cfg.display()
  
    # C: Load data ========================================================================================
           
    with tf.device('/CPU:0'):
        train_ds = load_ds(cfg, FLAGS.data_path, istrain = True)
        val_ds = load_ds(cfg, FLAGS.data_path, istrain = False)

    # D: Configure and fit model ===========================================================================

    model = DRSYolo(mode="training", cfg=cfg)      
    model.freeze_darknet_layers()

    #model.load_weights()

    model.compile()
    #model.summary()

    history = model.fit(train_ds, val_ds)
    
    """
    import tensorflow as tf
    from config import Config, Yolo, setattrs
    from methods.data import load_ds
    from methods.model import DRSYolo 

    cfg = Config() 
    setattrs(cfg, NAME = "default")            
    cfg.YOLO = Yolo(cfg, "./checkpoints/")     # ADJUST

    with tf.device('/CPU:0'):
        train_ds = load_ds(cfg, "./data/", istrain = True)
        val_ds = load_ds(cfg, "./data/", istrain = False)

    #x = list(train_ds.take(1))[0]    
    
    model = DRSYolo(mode="training", cfg=cfg)        
    #model.freeze_darknet_layers()
    #model.load_weights()

    #model.summary()
    model.compile()
    history = model.fit(train_ds, val_ds)
    """

    """OUT: MODEL.FIT

    model.compile()
    #model.summary()  
    # tf.keras.utils.plot_model(model.model)

    #model.model(x)
    tf.config.experimental_run_functions_eagerly(True)    
    history = model.fit(train_ds, val_ds)
    """
    
    """OUT: EAGER MODE

    # :note: purpose - debugging
    # :note: graph mode recommended for training
    
    tf.compat.v1.global_variables()
    [n for n in tf.compat.v1.get_default_graph().as_graph_def().node]

    from absl import logging

    avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

    for epoch in range(1, cfg.EPOCHS + 1):
        for batch, (batch_inputs, batch_outputs) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                total_loss = model.model(batch_inputs)
                
            grads = tape.gradient(total_loss, model.model.trainable_variables)

            optimizer = tf.keras.optimizers.Adam(lr=cfg.LR_INIT)
            optimizer.apply_gradients(zip(grads, model.model.trainable_variables))

            logging.info("{}_train_{}, {}".format(
                epoch, batch, total_loss.numpy()))
            avg_loss.update_state(total_loss)

        for batch, (batch_inputs, batch_outputs) in enumerate(val_ds):
            total_loss = model.model(batch_inputs)
            
            logging.info("{}_val_{}, {}, {}".format(
                epoch, batch, total_loss.numpy()))
            avg_val_loss.update_state(total_loss)

        logging.info("{}, train: {}, val: {}".format(
            epoch,
            avg_loss.result().numpy(),
            avg_val_loss.result().numpy()))

        avg_loss.reset_states()
        avg_val_loss.reset_states()
           
    """
    

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


# ADJUST: Apply automated class reweighting
# REFERENCE: flags.DEFINE_integer('weights_num_classes', None, 'specify num class for `weights` file if different, '
#                                 'useful in transfer learning with different number of classes')

