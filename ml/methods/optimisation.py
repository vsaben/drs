"""
    Description: Model optimisation
    Functions:
    - PART A: TFlite
    - PART B: Pruning
    - PART C: Quantisation
    - PART D: Weight Clustering
"""

import tensorflow as tf
from absl import app, flags, logging

# PART A: TFlite ===========================================================

def export_tflite(model_path):

    """Exports saved model to .tflite
    
    :note: extensions (tensorflow help)

    :param model_path: SavedModel directory

    :result: saved tflite model
    """

    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = model_path + '.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    logging.info('tflite model saved to {:s}'.format(tflite_path))

# PART B: Pruning ============================================================