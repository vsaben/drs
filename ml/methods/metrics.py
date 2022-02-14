"""
    Description: Model evaluation metrics and visualisations (in addition to losses)
    Function:
    - PART A: Confusion Matrix (CM)
    - PART B: AP
    - PART C: Original and annotated images; decision gradients
"""

import tensorflow as tf
import tensorflow.keras.layers as KL

import matplotlib
import matplotlib.pyplot as plt
import io
import itertools
import re
import textwrap
import numpy as np

from methods._data.reformat import pi, unpad, unbatch_dict, out_pose_to_cpos
from methods._data.targets import add_anchor_ids
from methods._model.head import assign_roi_level, _true

# PART A: Utilities =======================================================

# Source: https://github.com/tensorflow/tensorboard/issues/227

class FigureWriter:

    @classmethod
    def encode_fig(self, fig):
        
        """
        Converts a matplotlib figure ``fig`` into a TensorFlow Summary object
        that can be directly fed into ``Summary.FileWriter``.
        :param fig: A ``matplotlib.figure.Figure`` object.
        :return: A TensorFlow ``Summary`` protobuf object containing the plot image
                 as a image summary.
        """

        # attach a new canvas if not exists
        if fig.canvas is None:
            matplotlib.backends.backend_agg.FigureCanvasAgg(fig)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()

        # get PNG data from the figure
        png_buffer = io.BytesIO()
        fig.canvas.print_png(png_buffer)
        png_encoded = png_buffer.getvalue()
        png_buffer.close()

        # Convert png to tf image. Add batch dimension

        img = tf.image.decode_png(png_encoded, channels=4)
        img = tf.expand_dims(img, 0)

        return img


class PlotConfusionMatrix:
    
    """Plot confusion matrix to tensorboard"""

    def __init__(self, writer, confusion, step, lvl):

        """
        :param writer: cm_training / cm_validation
        :param classes: string class list
        :param confusion: confusion matrix
        """

        fig = self.plot_confusion(confusion)
        enc = FigureWriter.encode_fig(fig)

        suffix = "" if lvl is None else " ({:d})".format(lvl)

        with writer.as_default():
            tf.summary.image("Confusion Matrix" + suffix, enc, step=step)

    def plot_confusion(self, confusion):
        
        """Plot a confusion matrix

        :param cm: confusion matrix - [ncls, ncls] numpy array
    `   
        :return:  matplotlib.figure.Figure object        
        """

        nclass = np.shape(confusion)[0]
        classes = list(map(str, range(nclass)))

        fig = matplotlib.figure.Figure(figsize=(nclass, nclass), dpi=100, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(confusion, cmap='Oranges')

        #classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in classes]
        #classes = ['\n'.join(textwrap.wrap(l, 20)) for l in classes]

        tick_marks = np.arange(nclass)

        ax.set_xlabel('True', fontsize=10)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, ha='center', fontsize=8) # rotation = -90       
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()

        ax.set_ylabel('Predicted', fontsize=10)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, va='center', fontsize=8)
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()

        for i, j in itertools.product(range(nclass), range(nclass)):
            ax.text(j, i, int(confusion[i, j]), 
                    horizontalalignment="center", 
                    verticalalignment='center', 
                    color="black")
        fig.set_tight_layout(True)
        return fig

    

# PART B: Metric Layers =============================================================

# > 1: Classification

class ClsMetricLayer(KL.Layer):

    def __init__(self, cfg, name=None):
        super(ClsMetricLayer, self).__init__(name=name)
        self.cfg = cfg

    def call(self, inputs):

        """Log class losses as metrics, at all abstraction levels"""
        
        pd_boxes, gt_boxes = inputs
        nlevels = len(self.cfg['MASKS'])

        flat_cms = []

        # Per-level

        for lvl in range(nlevels):
            pcls, gcls = self.extract_single_layer_cls(pd_boxes[lvl], gt_boxes[lvl])

            cm = tf.math.confusion_matrix(gcls, pcls, 
                                          num_classes = self.cfg['NUM_CLASSES'], 
                                          name = 'cm_{:d}'.format(lvl))

            flat_cm = tf.cast(tf.reshape(cm, [-1]), tf.float32)
            self.add_cm_metrics(flat_cm, lvl)
            flat_cms.append(flat_cm)

        # Combined

        stack_cm = tf.stack(flat_cms)             # [nlevel, 4]
        flat_cm = tf.reduce_sum(stack_cm, axis=0) # [4]
        self.add_cm_metrics(flat_cm, -1)
     
        return flat_cm

    def extract_single_layer_cls(self, pbox, gbox):

        """Extracts predicted and ground-truth class values
        from yolo grids, at different scales
    
        :param pd_cls: [nbatch, gsize, gsize, plevel, [P(Class_1), ..., P(Class_n)]]
        :param gt_cls: [nbatch, gsize, gsize, plevel, cls integer]

        :result: [nbatch, ndetect, nclass + 1 (class prob + actual cls)]
        """

        _, _, pd_cls, _ = pbox
        _, gt_obj, gt_cls, _ = tf.split(gbox, (4, 1, 1, 10), axis=-1)
    
        obj_mask = tf.squeeze(gt_obj, -1)        # [nbatch, gsize, gsize, plevel]

        pcls_probs = tf.boolean_mask(pd_cls, obj_mask)                # [ndetect, ncls]
        pcls = tf.argmax(pcls_probs, axis=-1)                         # [ndetect]
        gcls = tf.squeeze(tf.boolean_mask(gt_cls, obj_mask), axis=-1) # [ndetect]

        return pcls, gcls

    def add_cm_metrics(self, flat_cm, level):

        suffix = "_{:d}".format(level) if level > -1 else "" 
        
        for i, short_name in enumerate(['tn', 'fn', 'fp', 'tp']):
            
            name = "{:s}".format(short_name) + suffix
            self.add_metric(flat_cm[i], name=name, aggregation='mean')

    def get_config(self):
        config = super(ClsMetricLayer, self).get_config() 
        config.update({'cfg': self.cfg})
        return config

# > 2: RPN

"""Compute AP across factors of interest:

    A. Overall
    B. Damage Status
    C. Visibility

"""

class AllMetricLayer(KL.Layer):
        
     def __init__(self, cfg, name=None):
        super(AllMetricLayer, self).__init__(name=name)
        self.cfg = cfg

        # Padded entries

        self.gt_rpn = self.create_init_variable([None, cfg['MAX_GT_INSTANCES'], 7], name='gt_rpn') 
        self.pd_rpn = self.create_init_variable([None, cfg['MAX_GT_INSTANCES'], 7], name='pd_rpn')

        self.gt_pose = self.create_init_variable([None, cfg['MAX_GT_INSTANCES'], 10], name='gt_pose')
        self.pd_pose = self.create_init_variable([None, cfg['MAX_GT_INSTANCES'], 10], name='pd_pose')
        self.pose_ids = self.create_init_variable([None, cfg['MAX_GT_INSTANCES'], 3], name='pose_ids')

        # Per image entries

        self.camera_rot = self.create_init_variable([None, 3])
        self.camera_R = self.create_init_variable([None, 3, 3])
        self.camera_w = self.create_init_variable([None, 1])
        self.camera_h = self.create_init_variable([None, 1])
        self.camera_fx = self.create_init_variable([None, 1]) 
        self.camera_fy = self.create_init_variable([None, 1])

     def create_init_variable(self, shape, name=None):

         var = tf.Variable(shape=shape,
                           initial_value = tf.zeros(shape=[1] + shape[1:], 
                                                    dtype=tf.float32),
                           validate_shape=False, 
                           trainable=False,
                           dtype = tf.float32, 
                           aggregation = tf.VariableAggregation.ONLY_FIRST_REPLICA, 
                           name=name)
         return var

     def call(self, inputs):
         pd_rpn, gt_rpn, features, pd_pose, gt_pose, cameras = inputs

         # Add metric IDs to rpn

         metric_ids = self.get_metric_ids(gt_rpn, features)
         gt_rpn = tf.concat([gt_rpn, metric_ids], axis=-1) # [lvl, vis]

         pd_lvl = self.add_lvl_ids(pd_rpn) 
         pd_rpn = tf.concat([pd_rpn, 
                             tf.expand_dims(pd_lvl, axis=-1)], axis=-1)

         self.pd_rpn.assign(pd_rpn)
         self.gt_rpn.assign(gt_rpn) 

         # Prepare pose information

         cls = tf.expand_dims(gt_rpn[..., 5], axis=-1)
         pose_ids = tf.concat([cls, metric_ids], axis=-1)
         self.pose_ids.assign(pose_ids)
         
         self.pd_pose.assign(pd_pose)
         self.gt_pose.assign(gt_pose)

         # Assign camera information

         self.camera_rot.assign(cameras['rot'])
         self.camera_R.assign(cameras['R'])
         self.camera_w.assign(cameras['w'])
         self.camera_h.assign(cameras['h'])
         self.camera_fx.assign(cameras['fx'])
         self.camera_fy.assign(cameras['fy'])

         return tf.constant(0)

     def add_lvl_ids(self, rpn):

         anchors = np.array(self.cfg['ANCHORS'], np.float32)           
         masks = self.cfg['MASKS']
         nlevels = len(masks)
         nscales = len(masks[0])

         anchor_id = add_anchor_ids(rpn, anchors, ids_only = _true)
         anchor_id = tf.ensure_shape(anchor_id, [None, self.cfg['MAX_GT_INSTANCES']])
         return assign_roi_level(anchor_id, nlevels, nscales)


     def get_metric_ids(self, gt_rpn, features):

        """Add metric ids associated with:

            1. Damage Status           [0, 1] >> 5 [Done]
            2. Prediction level [0, 1, 2]     >> 6
            3. Vehicle visibility [0, 1, 2]   >> 7
                0 (easy) - 1 (average) - 2 (hard)

        :param gt_rpn: [nbatch, cfg.MAX_GT_INSTANCES, [x1, y1, x2, y2, cls]]

        :result: [nbatch, cfg.MAX_GT_INSTANCES, [cls, lvl, vis]]
        """

        # Prediction level

        lvl = self.add_lvl_ids(gt_rpn)
        
        # Vehicle visibility

        occ = features[..., 0]
        tru = features[..., 1]

        easy = (tf.cast(tf.equal(lvl, 0), tf.int32) *  
                tf.cast(tf.equal(occ, 0), tf.int32) *
                tf.cast(tf.equal(tru, 0), tf.int32))
        
        medium = (tf.cast(tf.less_equal(lvl, 1), tf.int32) * 
                  tf.cast(tf.less_equal(occ, 0.08), tf.int32) *
                  tf.cast(tf.less_equal(tru, 1), tf.int32))

        difficulty = tf.stack([easy, medium, tf.ones_like(easy)], axis = -1)  
        vis = tf.cast(tf.argmax(difficulty, axis=-1), tf.float32)
        return tf.stack([lvl, vis], axis = -1)

     def get_config(self):
         config = super(AllMetricLayer, self).get_config() 
         config.update({'cfg': self.cfg})
         return config

class ComputeAP:

    """Modify default Mask R-CNN mAP calculation to 
        
        a. Aggregate precisions\recalls across images 
            (non-equal image AP contribution)
        b. Distinguish between features
        
        Process:
            1. Overlaps
            2. Matches
            3. Precision/recalls
            4. Calculate AP
    """

    IOU_THRESHES = tf.range(0.5, 1, 0.05, tf.float32)

    TYPES = {"undamaged":[0, 0],    # COL\VAL
             "damaged":  [0, 1],    # DAMAGE STATUS 
             "small":    [1, 0], 
             "medium":   [1, 1], 
             "large":    [1, 2],    # PREDICTION LEVEL       
             "easy":     [2, 0], 
             "average":  [2, 1],
             "hard":     [2, 2]}    # VEHICLE VISIBILITY
    
    def __init__(self, pd_rpn, gt_rpn):

        """Amalgamating function incorporating all AP calculation components
        
        :param pd_rpn: [bbox (4), score (1), cls (1), lvl (1)]
        :param gt_rpn: [bbox (4), cls (1), lvl (1), vis (1)]

        """

        self.TYPES_LST = list(self.TYPES.values())

        self.niou = len(self.IOU_THRESHES)
        self.ntyp = len(self.TYPES) + 1           # Add combined

        self.pd_rpn = pd_rpn
        self.gt_rpn = gt_rpn

        nbatch = tf.shape(pd_rpn)[0]      
        self.rpn = [(unpad(pd_rpn[b], 2), unpad(gt_rpn[b], 2)) for b in tf.range(nbatch)]

        self.result = self.compute_all_ap()

    def compute_all_ap(self):

        # 1. Overlaps (per batch)

        raw_scores = []
        overlaps = []

        for i, (p, g) in enumerate(self.rpn):

            g_bb, g_cls, g_lvl, g_vis = tf.split(g, (4, 1, 1, 1), axis=-1)
            p_bb, p_score, p_cls, p_lvl = tf.split(p, (4, 1, 1, 1), axis=-1)             # Ordered by score

            overlap = self.compute_overlaps(p_bb, g_bb)                           # [n_pd, n_gt] iou
            overlaps.append(overlap)

            n_pb = tf.size(p_score)
            score = tf.stack([tf.cast(tf.tile([i], [n_pb]), tf.float32), 
                              tf.range(n_pb, dtype = tf.float32),
                              tf.squeeze(p_score, axis=1)], axis = -1)
            raw_scores.append(score)

        unsorted_scores = tf.concat(raw_scores, axis=0)
        indices = tf.argsort(unsorted_scores[..., -1], direction = 'DESCENDING')
        scores = tf.gather(unsorted_scores, indices) # [image no, prediction no. (ordered by score), score]

        # 2: Matches

        n_pd = tf.shape(scores)[0]
        n_gt = tf.reduce_sum([tf.shape(ov)[-1] for ov in overlaps])

        if n_gt == 0:
            return -1 * tf.ones([self.ntyp, self.niou])

        if n_pd == 0:
            return tf.zeros([self.ntyp, self.niou])       
  
        pd_match, gt_match = self.compute_matches(n_pd, overlaps, scores) # [niou, n_pred], [[niou, n_gt] per image, ...]
                
        pd_ids = self.combine_pd_ids(scores)  # [n_pd, [cls_id, lvl_id, vis_id]]   (ordered by prediction score; predicted attributes)
        gt_ids = self.combine_gt_ids()        # [n_gt, [cls_id, lvl_id, vis_id]]   (in order of gt; same n_gt across niou)

        # 3. Average Precision 
        #    Check matches across initial threshold only (default = 50)
        #    If no associated predictions, return 0 AP
         
        if self.check_no_matches(gt_match):
            return tf.zeros([self.ntyp, self.niou])
       
        tot_ap = self.compute_ap_ious(pd_match, n_gt)                         # [niou]
        
        id_ap = tf.map_fn(lambda t: self.compute_ap_type(pd_match,            # [ntype - 1, niou]
                                                         pd_ids,
                                                         gt_match,
                                                         gt_ids,                                                         
                                                         self.TYPES_LST[t]), tf.range(len(self.TYPES_LST)), tf.float32)
        return tf.concat([tf.expand_dims(tot_ap, axis=0), id_ap], axis=0) # [ntype, niou]

    def compute_overlaps(self, boxes1, boxes2):
        """Computes IoU overlaps between two sets of boxes.
        boxes1, boxes2: [N, (x1, y1, x2, y2)].

        :result: iou overlaps [nbox1, nbox2]
        """

        # Box areas
    
        area1 = tf.expand_dims((boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]), axis=-1)
        area2 = tf.expand_dims((boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1]), axis=-1)

        # Compute overlaps to generate matrix of IOUs [nbox1, nbox2]

        boxes = tf.concat([boxes1, area1], axis = -1)
        cboxes = tf.concat([boxes2, area2], axis = -1)

        n1 = tf.shape(boxes)[0]
        n2 = tf.shape(cboxes)[0]

        overlaps = tf.zeros((n2, n1))    

        if tf.reduce_any(tf.shape(overlaps) == 0):
            return overlaps

        for i in tf.range(n2):
            cbox = cboxes[i]
            ious = self.compute_iou(cbox, boxes)       
            overlaps = tf.tensor_scatter_nd_update(overlaps, [[i]], [ious])
   
        return tf.transpose(overlaps)

    def compute_iou(self, cbox, boxes):
    
        """Calculates IoU of the given box with the array of the given boxes.
        :param box: [x1, y1, x2, y2, area]
        :param cboxes: [nbox, (x1, y1, x2, y2, area)] (comparison boxes) 

        :result: iou of box across comparison boxes
        """

        x1 = tf.maximum(cbox[0], boxes[:, 0])
        y1 = tf.maximum(cbox[1], boxes[:, 1])
        x2 = tf.minimum(cbox[2], boxes[:, 2])
        y2 = tf.minimum(cbox[3], boxes[:, 3])
        
        intersection = tf.nn.relu(x2 - x1) * tf.nn.relu(y2 - y1)
        union = cbox[4] + boxes[:, 4][:] - intersection[:]
        iou = intersection / union
        return iou

    def compute_matches(self, n_pd, overlaps, scores, score_thresh = 0):
        
        """Finds corresponding predicted and ground-truth detection

        :param n_pd: number of predictions (across all images)
        :param overlaps: [[n_pd, n_gt], ...]
        :param scores: [image no, prediction no. (ordered by score), score]

        :result: pd_match [niou, n_pd]
                 gt_match [[niou, n_gt per image], ...]
        """

        pd_match = -1 * tf.ones([self.niou, n_pd], tf.int32)                                  # Combined pd
        gt_match = [-1 * tf.ones([self.niou, tf.shape(ov)[-1]], tf.int32) for ov in overlaps] # Per image list of gt

        for score in scores:

            i = tf.cast(score[0], tf.int32) # Image no.
            j = tf.cast(score[1], tf.int32) # Prediction no. (in order)

            # > Sort matches by score

            rw_overlaps_i = overlaps[i]
            
            if tf.size(rw_overlaps_i) == 0:  # Move to next prediction if no ground-truths in image
                continue
            
            rw_overlaps = rw_overlaps_i[j]
            sorted_ixs = tf.argsort(rw_overlaps, axis=-1, direction = 'DESCENDING') 
            sorted_overlaps = tf.gather(rw_overlaps, sorted_ixs)

            # > Remove low scores

            low_score_idx = tf.cast(tf.where(sorted_overlaps < score_thresh), tf.int32) 
            if tf.size(low_score_idx) > 0: 
                sorted_ixs = sorted_ixs[:low_score_idx[0]]
  
            # > Find the match
            
            pd_cls = self.rpn[i][0][..., 5]
            gt_cls = self.rpn[i][1][..., 4]

            im_gmatch = gt_match[i]

            for l, iou_thresh in enumerate(self.IOU_THRESHES):

                iou_gmatch = im_gmatch[l]

                for k in sorted_ixs:  # Across image gt

                    if iou_gmatch[k] > -1:        # Skip IF gt already matched
                        continue                             

                    iou = rw_overlaps[k]  
                    if iou < iou_thresh:                        # Stop IF reach IoU smaller than the threshold
                        break                                           
            
                    if pd_cls[j] == gt_cls[k]:                  # Update IF match 
                        gt_match[i] = tf.tensor_scatter_nd_update(im_gmatch, [[l, k]], [j])
                        pd_match = tf.tensor_scatter_nd_update(pd_match, [[l, j]], [k])      # Give prediction no. (corresponds to scores)
                        break

        return pd_match, gt_match

    def combine_pd_ids(self, scores):

        """Combine pd_matches with predicted metric attributes for 
        easy component calculations.

        :param scores: [img_no, pred_no, scores] (ordered)
        :param pd_rpn: [nimage, cfg.MAX_GT_INSTANCES, [bbox (4), score (1), cls (1), lvl (1)]]

        :result: [n_pd, [cls_id, lvl_id, vis_id]] (ordered by score)
        """

        npd_ids = 2
        ids = []
        
        for img_no, pd_no, _ in scores:

            img_no = tf.cast(img_no, tf.int32)
            pd_no = tf.cast(pd_no, tf.int32)

            pred = self.pd_rpn[img_no][pd_no]

            id = tf.concat([pred[-npd_ids:], 
                           [pred[-1]]], # Use level in lieu of visibility 
                           axis = -1)            
            ids.append(id)

        return tf.stack(ids)

    def combine_gt_ids(self):
        return tf.concat([g[-3:] for p, g in self.rpn], axis = 0)

    def extract_id_matches(self, pd_match, pd_ids, gt_ids, type):

        """Extracts pd_match, n_gt corresponding to ground-truths
        matching ids:
        
            a. Damage: {0: undamaged, 
                        1: damaged} 
            b. Prediction Level: {0: small, 
                                  1: medium,
                                  2: large} (depends if yolo variant)
            c. Visibility: {0: easy, 
                            1: average, 
                            2: hard}

        :param pd_match: [niou, n_pd]
        :param pd_ids: [n_pd, [cls_id, lvl_id, vis_id]]
        :param gt_ids: [n_gt, [cls_id, lvl_id, vis_id]]
        :param type: [column: value] in ids matrix

        :result pd_match: filtered pd_match corresponding to id types
                          [niou, n_pd matching id]
        :result n_gt: number of ground-truths 
        """

        col, val = type

        pd_ix = tf.where(pd_ids[..., col] == val)
        pd_match_i = tf.transpose(tf.gather_nd(tf.transpose(pd_match), pd_ix))

        n_gt = tf.reduce_sum(tf.cast(tf.where(gt_ids[..., col] == val), tf.int32))
        return pd_match_i, n_gt
        
    def check_no_matches(self, gt_match):
        gt_match_t1 = tf.concat([g[0] for g in gt_match], axis=-1)
        return tf.reduce_all(tf.equal(gt_match_t1, -1))

    def compute_ap_type(self, pd_match, pd_ids, gt_match, gt_ids, type):

        pd_match_i, n_gt = self.extract_id_matches(pd_match, 
                                                   pd_ids,
                                                   gt_ids, 
                                                   type)
        
        if n_gt == 0:
            return -1 * tf.ones(self.niou)

        if self.check_no_matches(gt_match): 
            return tf.zeros(self.niou)
        
        return self.compute_ap_ious(pd_match_i, n_gt)

    def compute_ap(self, pd_match, n_gt):

        """Calculate AP. 
        
        :note: manage id levels outside

        :param pd_match: [gt no] corresponding to prediction
                         (possibly) filtered by gt id

        :result: ap
        """

        n_pd = tf.shape(pd_match)[0]

        if tf.reduce_all(tf.equal(pd_match, -1)):
            return 0.0

        precisions = tf.cumsum(tf.cast(pd_match > -1, tf.int32)) / tf.range(1, n_pd + 1)
        recalls = tf.cumsum(tf.cast(pd_match > -1, tf.int32)) / n_gt

        # Pad with start and end values to simplify the math
    
        precisions = tf.concat([[0], precisions, [0]], axis = 0)
        recalls = tf.concat([[0], recalls, [1]], axis = 0)

        # Ensure precision values decrease but don't increase. This way, the
        # precision value at each recall threshold is the maximum it can be
        # for all following recall thresholds, as specified by the VOC paper.

        for i in tf.range(tf.shape(precisions)[0] - 2, -1, -1):

            precisions = tf.tensor_scatter_nd_update(precisions, 
                                                     [[i]], 
                                                     [tf.maximum(precisions[i], precisions[i + 1])])

        # Compute mean AP over recall range
    
        recall_equ = tf.squeeze(tf.where(tf.logical_not(tf.equal(recalls[:-1], recalls[1:]))))
        indices = tf.cast(recall_equ, tf.int32) + 1

        ap = tf.reduce_sum((tf.gather(recalls, indices) - tf.gather(recalls, indices - 1)) * tf.gather(precisions, indices))
        ap = tf.cast(ap, tf.float32)
        return ap

    def compute_ap_ious(self, pd_match, n_gt):
        return tf.map_fn(lambda i: self.compute_ap(pd_match[i], n_gt), tf.range(self.niou), tf.float32) # [niou]

# > 3: 3D IoU =========================================================================

from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from methods._data.reformat import compute_bb3d_2d, DIM_KEYS

class Compute3DIOU:

    """Compute IoU across factors of interest (same as rpn)"""

    PANELS = {"front": ["ftr", "ftl", "fbl", "fbr"], 
              "rside": ["ftr", "fbr", "bbr", "btr"],
              "tside": ["ftr", "ftl", "btl", "btr"],
              "lside": ["ftl", "fbl", "bbl", "btl"],
              "bside": ["fbr", "fbl", "bbl", "bbr"], 
              "back": ["btr", "btl", "bbl", "bbr"]}

    TYPES_LST = list(ComputeAP.TYPES.values()) # Refer to ComputeAP for order / info

    def __init__(self, cameras, pd_pose, gt_pose, pose_ids, cfg):

        """Given perfect GT ROI, pose IoU is calculated based on 

            a. Projected masks
            b. Spherical intersection 

            across AP types

        :param cameras: batched camera dictionary
        :param pd_pose: [nbatch, cfg.MAX_GT_INSTANCES, [pos (3), dim (3), quart (4)]]
        :param gt_pose: [nbatch, cfg.MAX_GT_INSTANCES, [pos (3), dim (3), quart (4)]]
        :param pose_ids: [nbatch, cfg.MAX_GT_INSTANCES, [cls, lvl, vis]]
        :param cfg: configuration dictionary

        :result: [ndetect, [mask iou, lens iou]]
        """
        
        # Convert: Raw pose output >>> cpos 

        nbatch = tf.shape(pd_pose)[0]        
        cond = tf.where(gt_pose[..., 2] > 0)

        lst_gt_pose = self.extract_pose_lst(gt_pose, cond)
        lst_pd_pose = self.extract_pose_lst(pd_pose, cond)
        self.pose_ids = tf.gather_nd(pose_ids, cond)
        
        self.cfg = cfg
        unbatched_cams = unbatch_dict(cameras, nbatch)
        self.pd_pose = out_pose_to_cpos(unbatched_cams, lst_pd_pose, self.cfg, stack = True)
        self.gt_pose = out_pose_to_cpos(unbatched_cams, lst_gt_pose, self.cfg, stack = True) 

        self.cameras = [unbatched_cams[b] for b in cond[..., 0]]
        self.result = self.compute_iou_types()

    def extract_pose_lst(self, pose, cond):

        nbatch = tf.shape(pose)[0]
        comb_pose = tf.gather_nd(pose, cond)

        pose_lst = []
        for b in tf.range(nbatch, dtype=tf.int64):
            c = tf.where(cond[..., 0] == b)
            lst = tf.gather_nd(comb_pose, c)
            pose_lst.append(lst)

        return pose_lst

    def compute_iou_types(self):

        """Compute mask and lens 3D IoU across AP types.

        :param cameras: unbatched camera dictionary
        :param pd_pose: [ndetect, [pos, dim, rot]]
        :param gt_pose: [ndetect, [pos, dim, rot]]
        :param pose_ids: [ndetect, [cls, lvl, vis]]

        :result: [ndetect, [mask iou, lens iou]]
        """

        ndetect = tf.shape(self.pose_ids)[0]
        ious = tf.map_fn(lambda i: self.compute_detect_ious( # [ndetect, [mask iou, lens iou]]
            self.cameras[i], 
            self.pd_pose[i], 
            self.gt_pose[i]), 
                         tf.range(ndetect), 
                         dtype = tf.float32)


        all_iou = tf.reduce_mean(ious, axis=0, keepdims=True)
        typ_iou = tf.map_fn(lambda t: self.compute_iou_type(ious, self.TYPES_LST[t]), 
                            tf.range(len(self.TYPES_LST)), dtype=tf.float32)

        return tf.concat([all_iou, typ_iou], axis=0)

    def compute_iou_type(self, ious, type):
        
        col, val = type
        ix = (self.pose_ids[..., col] == val)       

        if not tf.reduce_any(ix):
            return -1 * tf.ones(2)
        
        sub_ious = tf.boolean_mask(ious, ix, axis = 0)
        return tf.reduce_mean(sub_ious, axis=0)

    def compute_detect_ious(self, camera, A, B):

        """Jointly calculate mask and lens iou types"""

        iou_mask = self.compute_iou_mask(camera, A, B) 
        iou_lens = self.compute_iou_lens(A, B)

        return tf.stack([iou_mask, iou_lens])

    def compute_iou_mask(self, camera, A, B):
 
        """Computes a novel 3D IoU approximating measure. 
        Extension of KITTI to 3 angles of orientation. Over-estimate
        of overlap. If projected overlap, 
    
        :param camera: {camera}
        :param A, B: [posx, posy, posz, 
                      dimx, dimy, dimz, 
                      rotx, roty, rotz]
    
        :result: 3D IoU
        """

        cornersA = compute_bb3d_2d(camera, A)
        cornersB = compute_bb3d_2d(camera, B)

        cA = dict(zip(DIM_KEYS, cornersA))
        cB = dict(zip(DIM_KEYS, cornersB))

        polA = self.get_polygon(cA) 
        polB = self.get_polygon(cB)

        inter = polA.intersection(polB).area      
        union = polA.union(polB).area 

        return inter / union

    def get_polygon(self, corners):

        """Extracts a polygon outlining a vehicle's shape in 2D. Projected 3D 
        vertices, associated with each side of a 6-sided 3D cuboid, are 
        transformed into polygons and combined (by union).
        """

        polygons = []
        for panel, vertices in self.PANELS.items():
       
            coords = [corners.get(vertex) for vertex in vertices]
            p = Polygon(coords).convex_hull
            polygons.append(p)

        return cascaded_union(polygons)

    def get_sphere_volume(self, r):
        return 4/3 * pi * r**3 

    def compute_iou_lens(self, A, B):

        c1, d1, _ = tf.split(A, (3, 3, 3), axis=-1) 
        c2, d2, _ = tf.split(B, (3, 3, 3), axis=-1)

        r1 = tf.reduce_max(d1)
        r2 = tf.reduce_max(d2)

        D = tf.norm(c1 - c2)

        # Check: No spherical intersection exists

        if D >= (r1 + r2):
            return 0.0

        vol1 = self.get_sphere_volume(r1)
        vol2 = self.get_sphere_volume(r2)

        # Check: Outer sphere contains inner sphere [ADJUST]

        if D <= tf.abs(r1 - r2):
            vmin = tf.reduce_min([vol1, vol2])
            vmax = tf.reduce_max([vol1, vol2])
            return vmin/vmax

        # Given intersection, analyse intersecting spherical caps

        # > Height

        theta1 = tf.acos((r1**2 + D**2 - r2**2)/(2*r1*D)) # cos rule
        theta2 = tf.acos((r2**2 + D**2 - r1**2)/(2*r2*D))

        h1 = r1 * (1 - tf.cos(theta1)) 
        h2 = r2 * (1 - tf.cos(theta2))

        # > Intersection

        inter_vol1 = (pi*h1**2)/3 * (3*r1 - h1)
        inter_vol2 = (pi*h2**2)/3 * (3*r2 - h2)
        inter = inter_vol1 + inter_vol2

        # > Union
        
        union = vol1 + vol2 - inter
        return inter / union

