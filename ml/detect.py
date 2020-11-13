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

from methods.data import load_tfrecord
from methods.collect import collect_detections
from methods.visualise import visualise_detections
from methods.model import DRSYolo
from config import Config

flags.DEFINE_string('name',         'default',             'configuration name')
flags.DEFINE_string('mode',         'detection',           'options: detection (output only), inference (output + loss)')
flags.DEFINE_string('model_dir',    './models/',           'storage directory for all models')
flags.DEFINE_string('example_dir',  './examples/',         'storage directory for all examples')
flags.DEFINE_string('tfrecord',     '*.tfrecord',          'regex for tfrecords containing image and camera metadata') 
flags.DEFINE_boolean('save',         False,                'option: whether to save annotated image')

def main(_argv):
    
    assert FLAGS.mode in ['detection', 'inference']

    model_dir = os.path.join(FLAGS.model_dir, FLAGS.name)
    cfg = Config.restore(model_dir)
    setattr(cfg, "MODEL_DIR", model_dir)

    cfg.display(FLAGS.mode)

    logging.info('model configuration restored')

    # PART A: Build model ================================

    cfg_mod = cfg.to_dict()

    model = DRSYolo(mode=FLAGS.mode, cfg=cfg_mod)
    model.build()
    
    model.summary() 

    # PART B: Load data ==================================
    
    regex = os.path.join(FLAGS.example_dir, FLAGS.tfrecord)

    t = time.time()
    ds = load_tfrecord(regex, FLAGS.mode, cfg)
   
    x, camera = ds.as_numpy_iterator().next()

    print('R: ', camera['R']) # ADJUST

    # PART C: Process outputs ============================
    
    nvalids, outs = model.predict(x)

    print(outs) # ADJUST

    nvalids = nvalids.numpy()
    logging.info('image detections: {} | time: {:.4f}'.format(nvalids, time.time() - t))

    for i, nvalid in enumerate(nvalids):

        logging.info('image: {:d} | detections: {:d}'.format(i, nvalid))        
        if nvalid == 0: continue
                        
        detections = collect_detections(outs)

        for i, det in enumerate(dets):
            
            ndam = sum(det[:, 4])
            logging.info('image: {:d} | detections: {:d} | damaged: {:d}'.format(i, ndetect, ndam))

            for j, d in enumerate(det):

                bbox, score, cls, pos, dim, rot = tf.split(d, (4, 1, 1, 3, 3, 3), axis=-1)

                update = '\t{:d} score: {:.2f} cls: {:s} bbox: {} pos: {} dim: {} rot: {}'.format(j,
                                                                                                 score, 
                                                                                                 cfg.CLASSES[int(cls)],
                                                                                                 np.round(pos.numpy(), 2).tolist(),
                                                                                                 np.round(pos.numpy(), 2).tolist(),                                                                                         np.round(dim.numpy(), 2).tolist(),
                                                                                                 np.round(rot.numpy(), 2).tolist())
                logging.info(update)
                                
    # PART D: Draw annotations =========================

        ann_img = visualise_detections(dets)
    
        if FLAGS.save:

            cv2.imwrite()
            logging.info('output saved to: {}'.format(FLAGS.output))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
