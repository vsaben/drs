"""
    Description: Perform wide-view aerial vehicle pose reconstruction and 
                    damage status determination for a single image. Camera 
                    properties assumed known. Annotated image includes 
                    ground truth information (if provided). 

    Function:
        - PART A: Build model
        - PART B: Load data
        - PART C: Process outputs
        - PART D: Draw annotations
"""


import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import os

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

from methods.data import load_detect_ds
from methods.visualise import get_ann_images
from methods.model import DRSYolo
from config import Config

flags.DEFINE_string('name',         'default',             'configuration name')
flags.DEFINE_string('model_dir',    './models/',           'storage directory for all models')
flags.DEFINE_string('example_dir',  './examples/',         'storage directory for all examples')
flags.DEFINE_string('tfrecord',     '*.tfrecord',          'regex for tfrecords containing image and camera metadata') 
flags.DEFINE_boolean('save',         False,                'option: whether to save annotated image')

def main(_argv):
    
    model_dir = os.path.join(FLAGS.model_dir, FLAGS.name)
    
    cfg = Config.restore(model_dir)
    setattr(cfg, "MODEL_DIR", model_dir)
    cfg.display("detection")

    logging.info('model configuration restored')

    # PART A: Build model ================================

    cfg_mod = cfg.to_dict()

    model = DRSYolo(mode="detection", cfg=cfg_mod)
    model.build()   
    model.summary() 

    # PART B: Load data and make prediction ===============
    
    regex = os.path.join(FLAGS.example_dir, FLAGS.tfrecord)

    t0 = time.time()
    ds = load_detect_ds(regex, cfg)
    tdata = time.time()

    ds = list(ds)[0]

    # PART C: Process outputs ============================
    
    padded_anns = model.predict(ds['input_image'])
    touts = time.time()

    print('PADDED_ANNS', padded_anns) # ADJUST

    nimage = tf.shape(padded_anns)[0]

    logging.info('images: {:d}'.format(nimage))
    logging.info('time intervals: {:.4f} | data {:.4f} | predict {:.4f}'.format(touts - t0, 
                                                                                tdata - t0, 
                                                                                touts - tdata))

    ann_images = get_ann_images(padded_anns, ds['camera'], ds['image_full'], cfg_mod)
    
    for i, img in enumerate(ann_images.numpy()):
        cv2.imshow('image', img.astype(np.uint8))
        cv2.waitKey()

    # PART D: Draw annotations =========================

    #    if FLAGS.save:
    #
    #        cv2.imwrite()
    #        logging.info('output saved to: {}'.format(FLAGS.output))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
