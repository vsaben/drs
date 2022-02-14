from absl import logging
import numpy as np
import tensorflow as tf
import cv2

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

YOLOV3_TINY_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
]

def load_darknet_weights(model, weights_file, tiny=False):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    if tiny:
        layers = YOLOV3_TINY_LAYER_LIST
    else:
        layers = YOLOV3_LAYER_LIST

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            logging.info("{}/{} {}".format(
                sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.get_input_shape_at(0)[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def broadcast_iou(pbox, gbox):

    """Calculates the IOU of predicted (pbox) to ground-truth boxes (gbox)

    :param pbox: (ngrid, ngrid, 3, (x1, y1, x2, y2))
    :param gbox: (N, (x1, y1, x2, y2))

    """

    # broadcast boxes
    pbox = tf.expand_dims(pbox, -2)
    gbox = tf.expand_dims(gbox, 0)

    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(pbox), tf.shape(gbox))
    pbox = tf.broadcast_to(pbox, new_shape)
    gbox = tf.broadcast_to(gbox, new_shape)

    int_w = tf.maximum(tf.minimum(pbox[..., 2], gbox[..., 2]) -
                       tf.maximum(pbox[..., 0], gbox[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(pbox[..., 3], gbox[..., 3]) -
                       tf.maximum(pbox[..., 1], gbox[..., 1]), 0)
    int_area = int_w * int_h
    pbox_area = (pbox[..., 2] - pbox[..., 0]) * (pbox[..., 3] - pbox[..., 1])
    gbox_area = (gbox[..., 2] - gbox[..., 0]) * (gbox[..., 3] - gbox[..., 1])
    return int_area / (pbox_area + gbox_area - int_area)

def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)

def trim_zeros_graph(boxes, name='trim_zeros'):
    """Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.
    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros

_false = tf.constant(False, tf.bool)
_true = tf.constant(True, tf.bool)

def GridToImg(tensor, anchors, isforward = _true):

    """Converts grid space to normalised image coordinates 
    (and vice versa) in grid form.
    
    :param tensor: single output layer predictions in grid or normalised coordinates
                   grid/img: [nbatch, grid, grid, 3, [x, y, w, h, ...]]
                  
    :param anchors: anchors associated with single output layer 
    :option isforward: grid to image (True)
                       image to grid (False)    
    """

    grid_size = tf.shape(tensor)[1]
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)   

    xy_t = tensor[..., 0:2]
    wh_t = tensor[..., 2:4]

    def true_fn():
        xy_img = (xy_t + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
        wh_img = tf.exp(wh_t) * anchors
        return tf.concat([xy_img, wh_img], axis=-1)

    def false_fn():
        xy_grd = xy_t * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)
        wh_grd = tf.math.log(wh_t / anchors)
        wh_grd = tf.where(tf.math.is_inf(wh_grd), tf.zeros_like(wh_grd), wh_grd)  
        return tf.concat([xy_grd, wh_grd], axis=-1)

    return tf.cond(isforward, true_fn, false_fn)

def CenterToMinMax(tensor, isforward = _true):

    """Converts centered (x, y, w, h) to min-max (x1, y1, x2, y2) coordinates 
    (and vice versa).

    :param tensor: [..., (x, y, w, h)]      (if centered)
                   [..., (x1, y1, x2, y2)]  (if min-max)

    :option isforward: (x, y, w, h) to (x1, y1, x2, y2) [True]
                       (x1, y1, x2, y2) to (x, y, w, h) [False]
    """

    xy_t = tensor[..., 0:2]
    tt_t = tensor[..., 2:4]

    def true_fn():
        box_min = xy_t - tt_t / 2
        box_max = xy_t + tt_t / 2
        return tf.concat([box_min, box_max], axis=-1)

    def false_fn():
        box_xy = (xy_t + tt_t) / 2
        box_wh = tt_t - xy_t
        return tf.concat([box_xy, box_wh], axis=-1)

    return tf.cond(isforward, true_fn, false_fn)