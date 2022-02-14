"""
    Description: Convert default yolo weights (from pjreddie.com) 
                 to model consistent tf checkpoints.
                 Adapted from yolo-tf2.
"""

from absl import logging

import numpy as np
import os

def load_yolo_weights(model):

    """Converts raw yolo weights to a tf checkpoint which is consistent with 
    the selected backbone. Saves tf checkpoint to model directory.

    :param model: model class (training mode)

    :result: path to saved tf checkpoint
    """

    # Extract yolo layers

    mod = model.model
    mod.summary() # REMOVE

    layer_size = model.cfg['LAYER_SIZE']
    output_pos = model.cfg['OUTPUT_POS']
    weights_file = model.cfg['DEFAULT_YOLO_WEIGHT_PATH']

    # Open weight file
    
    with open(weights_file, 'rb') as wf:
        
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
        
        j = 0
        for i in range(layer_size):
            
            conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
            bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'   

            conv_layer = mod.get_layer(conv_layer_name)
            filters = conv_layer.filters
            k_size = conv_layer.kernel_size[0]
            in_dim = conv_layer.input_shape[-1]
            
            if i not in output_pos:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                bn_layer = mod.get_layer(bn_layer_name)
                j += 1
            else:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

            logging.info("{}/{}".format(conv_layer_name, bn_layer_name if i not in output_pos else 'output'))

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))

            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if i not in output_pos:
                conv_layer.set_weights([conv_weights])
                bn_layer.set_weights(bn_weights)  
            else:
                conv_layer.set_weights([conv_weights, conv_bias]) 

        current = wf.tell()
        end = wf.seek(0, 2)

        logging.info('{:d} - {:d} = {:d} weights unused'.format(end, current, end - current))

        misc_dir = os.path.join(model.cfg['MODEL_DIR'], 'miscellaneous.txt')
        with open(misc_dir, 'a') as f:
            f.write('pretrained yolo weights used: {:d}/{:d}\n'.format(current, end))

        ckpt_dir = os.path.join(model.cfg['MODEL_DIR'], 'checkpoints', 'weights')
        mod.save_weights(ckpt_dir)

        logging.info('model weights saved to {:s}'.format(ckpt_dir))



