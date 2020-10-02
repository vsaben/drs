"""
    Description: Collect YOLO-base functions and perform base model selection
    Function:
    - PART A: Choose base YOLO model and associated cfg, head configuration
    - PART B: Collate model
    - PART C: Freeze specified darknet layers
"""

import datetime
import os

import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.layers as KL 
import tensorflow.keras.metrics as KM
from tensorflow.keras.regularizers import l2    

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, 
    EarlyStopping, 
    ModelCheckpoint, 
    TensorBoard
)

from methods._models_yolo import build_yolo_graph, extract_anchors_masks
from methods._model_head import build_drs_head
from methods._data_targets import transform_targets, extract_roi_pd
from methods.loss import compute_rpn_losses, compute_pose_losses


# B: Collate model ===========================================================================

class DRSYolo():
    """Encapsulates DRSYolo functionality"""

    def __init__(self, mode, cfg):

        """
        :param mode: "training" or "inference"
        :param cfg: configuration settings
        """

        assert mode in ['training', 'inference']
        self.mode = mode
        self.cfg = cfg
        self.model = self.build(mode=mode, cfg=cfg)

    def build(self, mode, cfg):

        """Build DRSYolo architecture
        
        :param: see init

        :input_image:     [nbatch, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.CHANNELS]
        :input_rpn:       [nbatch, cfg.MAX_GT_INSTANCES, bbox [4] + cls [1]] 
        :input_pose:      [nbatch, cfg.MAX_GT_INSTANCES, center [2] + depth [1] + dims [3] + quart [4]]       
        """

        # Inputs

        input_image = KL.Input([cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.CHANNELS], name='input_image')                
        inputs = [input_image] 

        if mode == 'training':     
            input_pose = KL.Input([cfg.MAX_GT_INSTANCES, 10], name='input_pose')   
            input_rpn = KL.Input([cfg.MAX_GT_INSTANCES, 5], name='input_rpn')                     
            inputs += [input_pose, input_rpn]

            gt_boxes = KL.Lambda(lambda x: transform_targets(*x, cfg=cfg), name='gt_boxes')([input_rpn, input_pose])

        # Model
        
        rpn_fmaps, pd_boxes, nms = build_yolo_graph(input_image, cfg) 
              
        #if mode == "training" and not cfg.USE_RPN_ROI: 
        #rpn_rois = input_all_pad[:, :, 0:4]

        roi = input_rpn[..., :4]                                         # [nbatch, cfg.MAX_GT_INSTANCES, (x1, y1, x2, y2)]
        pd_pose = build_drs_head(roi, rpn_fmaps, cfg)                           # [nbatch, cfg.MAX_GT_INSTANCES, 1, 10]
        gt_pose = input_pose                                            # [nbatch, cfg.MAX_GT_INSTANCES, 10]
 
        # Losses
        
        rpn_loss = RpnMetricLayer(cfg, name='rpn_loss')([pd_boxes, gt_boxes])
        pose_loss = PoseMetricLayer(cfg, name='pose_loss')([pd_pose, gt_pose])

        outputs = [rpn_loss, pose_loss] 

        return Model(inputs, outputs, name='drs_yolo')

    def load_weights(self):

        """Load saved model weights OR intialise default yolo backend 
        :note: weight transfer assumed better than weight initialisation strategies.
            COCO initialised YOLO already capable of identifying vehicle types

        :param weight_path: path to model checkpoints
                            default coco-trained yolo weights OR 
                            user-specified full-model weight file

        :result: model weights initialised
        """

        weight_path = self.cfg.YOLO.CKPT

        file_name = os.path.basename(os.path.normpath(weight_path))
        by_name = (file_name == "model.ckpt")
        self.model.load_weights(weight_path, by_name = by_name)
        
    
    def freeze_darknet_layers(self, isper = True):

        """Unfreeze darknet conv2d, batch-norm pairs assorted from back to front
        :note: darknet submodel initially frozen, then thawed

        :param model: full drs model
        :param no_unfreeze: number/percentage of darknet layers to thaw 
        :option isper: whether no_unfreeze is a percentage

        :result: modified darknet model
                 number of layers unfrozen
        """

        no_unfreeze = self.cfg.PER_DARKNET_UNFROZEN
        darknet = self.model.get_layer('yolo').get_layer('darknet')
    
        # Freeze darknet

        if no_unfreeze == 0:
            darknet.trainable = False
            return

        trainable_layers = [layer for layer in darknet.layers if 
                            ('conv2d' in layer.name) or  
                            ('batch_norm' in layer.name)]
        no_trainable = len(trainable_layers)

        # Unfrozen darknet

        if isper: no_unfreeze = round(no_unfreeze * no_trainable)
        no_unfreeze_layers = no_unfreeze * 2
        if no_unfreeze_layers >= no_trainable: return

        # Partially frozen darknet 

        darknet.trainable = False
        for layer in trainable_layers[-no_unfreeze_layers:]:
            layer.trainable = True
    
    def compile(self):

        """Compiles model object. Adds losses, metrics and regularisation prior 
        to compilation. Disables training of batch normalisation layers (if specified)
        """

        cfg = self.cfg
        optimizer = tf.keras.optimizers.Adam(lr=cfg.LR_INIT)

        # Add: Losses

        #self.model._losses = []             ADJUST
        #self.model._per_input_losses = {}
   
        for loss_name, loss_weight in cfg.LOSSES.items():
            layer = self.model.get_layer(loss_name)
            #if (layer.output in model.losses): 
            #    continue
            loss = tf.reduce_mean(layer.output, keepdims=True) * loss_weight 
            self.model.add_loss(loss) 

        # Add: L2 Regularisation
        # note: skip gamma and beta weights of batch normalization layers.
        # ADJUST: division by size

        reg_losses = [l2(cfg.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                        for w in self.model.trainable_weights
                        if 'gamma' not in w.name and 'beta' not in w.name]
        
        tot_reg_losses = tf.add_n(reg_losses)
        self.model.add_loss(lambda: tot_reg_losses)
        # self.model.add_metric(tot_reg_losses, aggregation='mean', name='reg_loss')

        self.model.compile(optimizer=optimizer, 
                           loss=[None]*len(self.model.outputs))
                      
        # Add: Metrics

        # > RPN (ADJUST: Convert function for rpn/head)

        #for subloss_name in {**cfg.RPN_TYPE_LOSSES}: # ,**cfg.METRICS, , subloss_weight .items()
            #if (loss_name in model.metrics_names) or (loss_weight == 0):
            #    continue
        #    layer = self.model.get_layer(subloss_name) 
        #    subloss = tf.reduce_mean(layer.output, keepdims=True)
        #    print(subloss, layer.output) # ADJUST
        #    self.model.add_metric(KM.Mean()(subloss), name=subloss_name) 

        #   for metric_name in metric_names:
        #        layer = model.get_layer(metric_name)             
        #               # reduced_sum >>> reduced_mean [ADJUST]
        #        

        # Disable: Training of batch normalisation layers

        if not cfg.TRAIN_BN:
            yolo_model = self.model.get_layer('yolo')
            bn_layers = [l for l in self.model.layers if ('batch_normalization' in l.name)]
            for bn_layer in bn_layers:
                bn_layer.trainable = False


    def get_callbacks(self):        
        cfg = self.cfg

        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")    
        log_dir = os.path.join("logs", cfg.NAME, time)            
        ckpt_dir = os.path.join("checkpoints", cfg.NAME)    
   
        callbacks = [
                ReduceLROnPlateau(verbose=1, min_lr = cfg.LR_END),
                EarlyStopping(patience=3, verbose=1),            
                TensorBoard(log_dir=log_dir, 
                            histogram_freq=1, 
                            write_images=True), 
                ModelCheckpoint(ckpt_dir + '{epoch}.ckpt', 
                                verbose=1, 
                                save_best_only=True,
                                save_weights_only=True, 
                                save_freq = 'epoch')]

        return callbacks

    def summary(self):
        self.model.summary()

    def fit(self, train_ds, val_ds):
        cfg = self.cfg
        self.model.fit(train_ds, 
                       epochs=cfg.EPOCHS,
                       #callbacks=self.get_callbacks(),
                       validation_data=val_ds, 
                       verbose=2) 

class RpnMetricLayer(KL.Layer):   
    
    def __init__(self, cfg, name=None):
        super(RpnMetricLayer, self).__init__(name=name)
        self.cfg = cfg

    def call(self, inputs):
        """ Log all computed rpn losses as metrics, at all levels of abstraction incl.
        type, level and type-level.
        
        :note: 'aggregation' = how to aggregate the per-batch values over each epoch
        
        :return: batch rpn loss
        """
        
        pd_boxes, gt_boxes = inputs
        all_rpn_losses = KL.Lambda(lambda x: compute_rpn_losses(*x, cfg=self.cfg))([pd_boxes, gt_boxes])

        anchors, masks = extract_anchors_masks(self.cfg.YOLO)
        nlevels = len(masks)

        # Type [t]

        lweights = tf.constant(list(self.cfg.RPN_LVL_LOSSES.values())[:nlevels])
        all_tloss = tf.transpose(all_rpn_losses, [0, 2, 1])             # [nbatch, nlosses, nlevels]        
        weighted_tloss = all_tloss * lweights                           # [nbatch, nlosses, nlevels]        
        sum_tloss = tf.reduce_sum(weighted_tloss, axis = 2)             # [nbatch, nlosses]        
        mean_tloss = tf.reduce_mean(sum_tloss, axis = 0)                # [nlosses]
        
        tnames = list(self.cfg.RPN_TYPE_LOSSES.keys())
        ntypes = len(tnames)
        for i in range(ntypes):
            self.add_metric(mean_tloss[i], name=tnames[i], aggregation="mean") 
        
        # Level [l]

        tweights = tf.constant(list(self.cfg.RPN_TYPE_LOSSES.values()))
        weighted_lloss = all_rpn_losses * tweights                     # [nbatch, nlevels, nlosses]
        sum_lloss = tf.reduce_sum(weighted_lloss, axis = 2)            # [nbatch, nlevels]
        mean_lloss = tf.reduce_mean(sum_lloss, axis = 0)               # [nlevels]

        lnames = list(self.cfg.RPN_LVL_LOSSES.keys())[:nlevels]
        for i in range(nlevels):
            self.add_metric(mean_lloss[i], name=lnames[i], aggregation="mean") 

        # Type-Level

        for lvl in range(nlevels):
            for typ in range(ntypes): 
                mean_tlloss = tf.reduce_mean(all_rpn_losses[:, lvl, typ])                
                lname = extract_subloss_shortname(lnames[lvl])
                tname = extract_subloss_shortname(tnames[typ])                
                tlname = 'rpn_{:s}_{:s}_loss'.format(lname, tname) 
                self.add_metric(mean_tlloss, name=tlname, aggregation="mean")

        # Final
                
        rpn_loss = tf.reduce_sum(mean_tloss * tweights, name='rpn_loss')
        return rpn_loss

def extract_subloss_shortname(name):

    _1 = name.find('_') + 1
    _2 = name.rfind('_')

    return name[_1:_2]

class PoseMetricLayer(KL.Layer):
    
    def __init__(self, cfg, name=None):
        super(PoseMetricLayer, self).__init__(name=name)
        self.cfg = cfg

    def call(self, inputs):
        """Log all computed pose losses as metrics including 
        bbox (optional), class (optional), center, depth, dimension and rotation
        
        :param pd_head:
        :param gt_head:

        """
        pd_pose, gt_pose = inputs
        all_pose_losses = KL.Lambda(lambda x: compute_pose_losses(*x, cfg=self.cfg))([pd_pose, gt_pose]) # [nbatch, nlosses]

        pose_dict = self.cfg.POSE_SUBLOSSES
        nlosses = len(pose_dict)
        pnames = list(pose_dict.keys())
        pweights = list(pose_dict.values())

        # Sublosses

        for i in range(nlosses):
            self.add_metric(all_pose_losses[i], name=pnames[i], aggregation='mean')

        # Final

        pose_loss = tf.reduce_sum(all_pose_losses * pweights, name='pose_loss')
        return pose_loss

#def depad(tensor):
#    ragged = tf.RaggedTensor.from_tensor(tensor, padding=0, ragged_rank = 2)
#    nested_row_lengths = ragged.nested_row_lengths()
#    ix = tf.squeeze(tf.where(tf.not_equal(row_lengths, 0)))
#    return tf.gather(ragged, ix)    
#    #return tensor.to_tensor()

