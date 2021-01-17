"""
    Description: Model evaluation metrics and visualisations (in addition to losses)
    Function:
    - PART A: Confusion Matrix (CM)
    - PART B: mAP
    - PART C: Original and annotated images; decision gradients
"""

import tensorflow as tf

import matplotlib
import io
import itertools
import re
import textwrap
import numpy as np

# PART A: Confusion Matrix =======================================================

# Source: https://github.com/tensorflow/tensorboard/issues/227

class PlotConfusionMatrix:
    
    """Plot confusion matrix to tensorboard"""

    def __init__(self, writer, confusion, step, lvl):

        """
        :param writer: cm_training / cm_validation
        :param classes: string class list
        :param confusion: confusion matrix
        """

        fig = self.plot_confusion(confusion)
        enc = self.encode_fig(fig)

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

# PART B: mAP ======================================================================= 

"""Compute mAP across factors of interest:

    A. Overall
    B. Object Size: S, M, L
    C. 

"""

def compute_matches(gt_boxes, gt_class_ids, gt_masks,
                    pred_boxes, pred_class_ids, pred_scores, pred_masks,
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps


def compute_ap(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps


def compute_ap_range(gt_box, gt_class_id, gt_mask,
                     pred_box, pred_class_id, pred_score, pred_mask,
                     iou_thresholds=None, verbose=1):
    """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)
    
    # Compute AP over range of IoU thresholds
    AP = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls, overlaps =\
            compute_ap(gt_box, gt_class_id, gt_mask,
                        pred_box, pred_class_id, pred_score, pred_mask,
                        iou_threshold=iou_threshold)
        if verbose:
            print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))
        AP.append(ap)
    AP = np.array(AP).mean()
    if verbose:
        print("AP @{:.2f}-{:.2f}:\t {:.3f}".format(
            iou_thresholds[0], iou_thresholds[-1], AP))
    return AP


def compute_recall(pred_boxes, gt_boxes, iou):
    """Compute the recall at the given IoU threshold. It's an indication
    of how many GT boxes were found by the given prediction boxes.

    pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    gt_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    """
    # Measure overlaps
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)
    positive_ids = np.where(iou_max >= iou)[0]
    matched_gt_boxes = iou_argmax[positive_ids]

    recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]
    return recall, positive_ids

# PART C: Validation images (original, annotated, decision gradients) ====================

class PlotValImages:

    def plot_val(self, val_ds, nimage = 3):
       
        """Writes a sample of validation images 
        including their
   
            a. predicted and ground-truth annotations
            b. decision gradents

        to the tensorboard"""

        val_ds = list(val_ds
                        .unbatch()
                        .take(nimage)
                        .batch(nimage)
                        .take(1))[0][0]

        writer = os.path.join(self.cfg['LOG_DIR'], "sample")
         
        with writer.as_default():

            original = val_ds['input_image']

            outs = self.model.predict(original)
            
            #p_annotated = 
            #o_annotated =
           
            #tf.summary.image('Sample Batch: Validation Data', images, max_outputs=nimage, step=0)

                # Decision Gradients

            tf.summary.image('Sample: Validation Data', images, max_outputs=nimage, step=0)