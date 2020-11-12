"""
    Description: Convert default yolo weights (from pjreddie.com) 
                 to model consistent tf checkpoints.
                 Adapted from yolo-tf2.
"""

from absl import logging

import numpy as np

import tensorflow as tf
import os

def load_raw_weights(model):

    """Converts raw yolo weights to a tf checkpoint which is consistent with 
    the selected backbone. Saves tf checkpoint to model directory.

    :param model: model class (training mode)

    :result: path to saved tf checkpoint
    """

    # Extract yolo layers

    mod = model.model
    submods = mod.get_layer('yolo').layers[1:] # exclude input

    # Open weight file
  
    weights_file = model.cfg['DEFAULT_YOLO_WEIGHT_PATH']

    with open(weights_file, 'rb') as wf:
        
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
        
        for submod in submods:
            layers = submod.layers
            nlayers = len(layers)

            for i, layer in enumerate(layers):

                if not layer.name.startswith('conv2d'): 
                    continue

                bnorm = None
                if i + 1 < nlayers and layers[i + 1].name.startswith('batch_norm'):
                    bnorm = layers[i + 1]

                logging.info("{}/{} {}".format(submod.name, layer.name, 'bn' if bnorm else 'bias'))

                filters = layer.filters
                size = layer.kernel_size[0]
                in_dim = layer.get_input_shape_at(0)[-1]

                if bnorm is None:
                    conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
                else:
                    # darknet [beta, gamma, mean, variance]
                    bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                    # tf [gamma, beta, mean, variance]
                    bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

                # darknet shape (out_dim, in_dim, height, width)
                conv_shape = (filters, in_dim, size, size)
                conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))

                # tf shape (height, width, in_dim, out_dim)
                conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

                if bnorm is None:
                    layer.set_weights([conv_weights, conv_bias])  
                else:
                    layer.set_weights([conv_weights])
                    bnorm.set_weights(bn_weights)

        current = wf.tell()
        end = wf.seek(0, 2)

        logging.info('{:d} - {:d} = {:d} weights unused'.format(end, current, end - current))

        misc_dir = os.path.join(model.cfg['MODEL_DIR'], 'miscellaneous.txt')
        with open(misc_dir, 'a') as f:
            f.write('pretrained yolo weights used: {:d}/{:d}\n'.format(current, end))

        ckpt_dir = os.path.join(model.cfg['MODEL_DIR'], 'checkpoints', 'weights')
        mod.save_weights(ckpt_dir)

        logging.info('model weights saved to {:s}'.format(ckpt_dir))



