"""
    Description: Collect YOLO-base functions and perform base model selection
    Function:
    - PART A: Choose base YOLO model and associated cfg, head configuration
    - PART B: Collate model
    - PART C: Freeze specified darknet layers
"""

import os
import numpy as np
import pickle

from absl import logging

import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as KL 
import tensorflow.keras.metrics as KM
import tensorflow.keras.regularizers as KR    
import tensorflow.keras.callbacks as KC

from methods._model.yolo import build_yolo_graph
from methods._model.head import RoiAlign, PoseGraph, DetectionTargetsLayer, OutLayer
from methods._data.targets import transform_targets
from methods._data import convert
from methods.loss import ClsMetricLayer, RpnMetricLayer, PoseMetricLayer, CombineLossesLayer
from methods.metrics import PlotConfusionMatrix

# B: Collate model ===========================================================================

class DRSYolo():

    """Encapsulates DRSYolo functionality"""

    def __init__(self, mode, cfg, val_ds=None):

        """
        :param mode: "training", "inference" or "detection"
        :param cfg: configuration settings
        """

        assert mode in ['training', 'inference', 'detection']
        self.mode = mode
        self.cfg = cfg
        self.val_ds = val_ds

    def build_model(self):

        """Build DRSYolo architecture
        
        :param: see init

        :input_image:     [nbatch, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.CHANNELS]
        :input_rpn:       [nbatch, cfg.MAX_GT_INSTANCES, bbox [4] + cls [1]] 
        :input_pose:      [nbatch, cfg.MAX_GT_INSTANCES, center [2] + depth [1] + dims [3] + quart [4]]       
        """

        cfg = self.cfg

        # Inputs

        input_image = KL.Input([cfg['IMAGE_SIZE'], cfg['IMAGE_SIZE'], cfg['CHANNELS']], name='input_image')                

        if self.mode == 'training':     
            input_pose = KL.Input([cfg['MAX_GT_INSTANCES'], 10], name='input_pose')   
            input_rpn = KL.Input([cfg['MAX_GT_INSTANCES'], 5], name='input_rpn')                     
            inputs = [input_image, input_pose, input_rpn]

            gt_boxes = KL.Lambda(lambda x: transform_targets(*x, cfg=cfg), name='gt_boxes')([input_rpn, input_pose])
            gt_od = input_rpn
            gt_pose = tf.concat([input_rpn, input_pose], axis = -1, name='gt_pose')  # [nbatch, cfg.MAX_GT_INSTANCES, 5 + 10]
            
        # RPN
        
        rpn_fmaps, pd_boxes, nms = build_yolo_graph(input_image, cfg)    
        rpn_roi = nms[0]

        if self.mode == "training":

        #    if cfg.USE_RPN_ROI:            
        #        gt_pose, gt_od = DetectionTargetsLayer(cfg, name = 'extract_pose_od')([rpn_roi, gt_boxes])
        #    else:
            
            rpn_roi = gt_od[..., :4]                     # [nbatch, cfg.MAX_GT_INSTANCES, (x1, y1, x2, y2)] 

        roi_align = RoiAlign(cfg, name='roi_align')([rpn_roi, rpn_fmaps])         
        
        # HEAD

        pd_pose = PoseGraph(cfg, name='pose_graph')([roi_align])        # [nbatch, cfg.MAX_GT_INSTANCES, 10] 
                      
        if self.mode == 'detection': 
            outs = OutLayer(name='out')([nms, pd_pose])
            return K.Model(input_image, outs, name='drs_yolo')
        
        # METRICS

        cls_metric = ClsMetricLayer(cfg, name='cm_metric')([pd_boxes, gt_boxes])

        # LOSSES

        # note: only training mode left
        # note: ensure model json serialisable: pruning, quantisation

        rpn_loss = RpnMetricLayer(cfg, name='rpn_loss')([pd_boxes, gt_boxes])        
        pose_loss = PoseMetricLayer(cfg, name='pose_loss')([pd_pose, gt_pose]) 

        loss = CombineLossesLayer(cfg, name='loss')([rpn_loss, pose_loss, cls_metric])
  
        return K.Model(inputs, loss, name='drs_yolo') 

    def build(self, transfer=False):
        self.model = self.build_model()
        self.compile()

        if self.mode == 'training' and not transfer:
            self.freeze_darknet_layers()
                    
        self.load_weights(optimiser=not transfer)
        

    def load_weights(self, optimiser=True):

        """Load saved model weights OR intialise default yolo backend 
        :note: weight transfer assumed better than weight initialisation strategies.
            COCO initialised YOLO already capable of identifying vehicle types

        :param weight_path: path to model checkpoints
                            default coco-trained yolo weights OR 
                            user-specified full-model weight file

        :result: model weights initialised
        """

        ckpt_dir = os.path.join(self.cfg['MODEL_DIR'], "checkpoints")
        weight_path = os.path.join(ckpt_dir, "weights")

        if self.mode == "training":
            if os.path.isdir(ckpt_dir):
                if optimiser:
                    optim_path = os.path.join(ckpt_dir, "optimiser.pkl") 
                    if os.path.isfile(optim_path):                  # important: set model weights after optimizer (otherwise reset)
                        with open(optim_path, 'rb') as f:
                            weights = pickle.load(f)
                
                        opt = tf.keras.optimizers.get('Adam') 
                        opt.set_weights(weights)
                        logging.info('optimiser weights loaded')    

                self.model.load_weights(weight_path, by_name=False) # by_name=False --> loading from tensorflow format topology            
                logging.info('model weights loaded: {:s}'.format(self.mode))            
            else:
                convert.load_raw_weights(self)
                logging.info('pretrained weights loaded: {:s}'.format(self.mode))
        else:
            train = DRSYolo("training", self.cfg)
            train.build(transfer=True)
            train.model.load_weights(weight_path, by_name=False)

            weighted_names = [submod.name for submod in train.model.layers if 
                              len(submod.get_weights()) > 0]

            for name in weighted_names:
                weights = train.model.get_layer(name).get_weights()            
                self.model.get_layer(name).set_weights(weights)

            logging.info('model weights loaded: {:s}'.format(self.mode))       

    @staticmethod
    def restore(cfg, val_ds):
        
        """Restores DRSYolo class during training

        :note: tensorflow bug in full model restore
               AttributeError: '_UserObject' object has no attribute 'summary'
               load model and optimiser weights (in the interim)
        """

        full_model_dir = os.path.join(cfg['MODEL_DIR'], 'model')
        cls = DRSYolo("training", cfg, val_ds)
        #cls.model = K.models.load(full_model_dir)
        return cls

    def freeze_darknet_layers(self, isper = True):

        """Unfreeze darknet conv2d, batch-norm pairs assorted from back to front
        :note: darknet submodel initially frozen, then thawed

        :param model: full drs model
        :param no_unfreeze: number/percentage of darknet layers to thaw 
        :option isper: whether no_unfreeze is a percentage

        :result: modified darknet model
                 number of layers unfrozen
        """

        no_unfreeze = self.cfg['PER_DARKNET_UNFROZEN']
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
        optimizer = K.optimizers.Adam(learning_rate=cfg['LR_INIT'])

        # Add: Regularisation loss [L2]
        # note: skip gamma and beta weights of batch normalization layers.
        # ADJUST: division by size

        reg_losses = [KR.l2(cfg['WEIGHT_DECAY'])(w) / tf.cast(tf.size(w), tf.float32)
                        for w in self.model.trainable_weights
                        if 'gamma' not in w.name and 'beta' not in w.name]
        
        tot_reg_losses = tf.add_n(reg_losses, 'reg_losses')
        self.model.add_loss(lambda: tot_reg_losses)

        self.model.compile(optimizer=optimizer,
                           loss=[None]*len(self.model.outputs))
                           

        # Disable: Training of batch normalisation layers

        if not cfg['TRAIN_BN']:
            yolo_model = self.model.get_layer('yolo')
            bn_layers = [l for l in self.model.layers if ('batch_normalization' in l.name)]
            for bn_layer in bn_layers:
                bn_layer.trainable = False


    def get_callbacks(self):        
        
        train_dir = os.path.join(self.cfg['LOG_DIR'], 'cm_train')
        train_fwriter = tf.summary.create_file_writer(train_dir)
        
        val_dir = os.path.join(self.cfg['LOG_DIR'], 'cm_validation')
        val_fwriter = tf.summary.create_file_writer(val_dir)
        
        ckpt_dir = os.path.join(self.cfg['MODEL_DIR'], "checkpoints")    
   
        callbacks = [
                KC.ReduceLROnPlateau(patience=10, verbose=2, min_lr = self.cfg['LR_END']),
                KC.EarlyStopping(patience=10, verbose=2),            
                KC.TensorBoard(log_dir=self.cfg['LOG_DIR'], 
                               histogram_freq=5, 
                               write_images=True,
                               write_graph=True, 
                               update_freq='epoch'), 
                ModelRestartCheckpoint(ckpt_dir, self.cfg['BEST_VAL_LOSS']), 
                EpochCM(train_fwriter, 
                        val_fwriter, 
                        self.cfg['TOTAL_EPOCHS'], 
                        len(self.cfg['MASKS']))]
        return callbacks

    def summary(self):
        self.model.summary()

    def fit(self, train_ds, val_ds):

        initial_epoch = self.cfg['TOTAL_EPOCHS']
        epochs = self.cfg['TOTAL_EPOCHS'] + self.cfg['EPOCHS']

        history = self.model.fit(train_ds, 
                                initial_epoch=initial_epoch, 
                                epochs=epochs,
                                callbacks=self.get_callbacks(),
                                validation_data=val_ds, 
                                verbose=2)
        return history

    def save(self, full=False, schematic=False):

        """Saves full model and optimiser weights (workaround)

        :note: optimiser weights saved separately | tensorflow
               error in model restore (revert when error resolved)
        """

        # Full model

        if full:
            full_model_dir = os.path.join(self.cfg['MODEL_DIR'], 'model')        
            self.model.save(full_model_dir) 
            logging.info('model saved to {:s}'.format(full_model_dir))

        # Optimiser

        optim_path = os.path.join(self.cfg['MODEL_DIR'], 'checkpoints', 'optimiser.pkl')        
        weights = tf.keras.optimizers.get('Adam').get_weights()

        with open(optim_path, 'wb') as f:
            pickle.dump(weights, f)

        logging.info('optimiser weights saved to {:s}'.format(optim_path))

        if schematic:

            """Plots model schematic to file"""

            plot_path = os.path.join(self.cfg['MODEL_DIR'], "model_schematic.png")

            self.model._layers = [layer for layer in self.model._layers if isinstance(layer, KL.Layer)]
            tf.keras.utils.plot_model(self.model, 
                                      to_file=plot_path, 
                                      show_layer_names=False, 
                                      expand_nested=True)
            
            logging.info("model schematic saved to {:s}".format(plot_path))       

    def predict(self, x):
        return self.model(x, training=False)

class ModelRestartCheckpoint(KC.Callback):    
    
    """Saves model weights of the best model, as defined by that that 
    yields the smallest 'val_loss' measure. Weights are saved to the
    checkpoint directory. The initial 'val_loss' is set to the 
    previously trained model's best 'val_loss'.
    """

    def __init__(self, ckpt_dir, initial_val_loss):
        super(ModelRestartCheckpoint, self).__init__()
        self.ckpt_dir = os.path.join(ckpt_dir, 'weights')
        self.initial_val_loss = initial_val_loss 
    
    def on_train_begin(self, logs=None):
        self.best = self.initial_val_loss

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_loss")
        if np.less(current, self.best):
            logging.info('val loss improved from {:.0f} to {:.0f}. weights saved to {:s}'
                            .format(self.best, current, self.ckpt_dir))
            self.best = current
            self.model.save_weights(self.ckpt_dir)
        else:
            logging.info('val loss did not improve.')
            
class EpochCM(KC.Callback):

    """Posts confusion matrix metrics (at each prediction-levels and overall)
    to the tensorboard. Confusion values are calculated over an entire epoch. This 
    avoids issues arising from undefined operations in tensorflow's batch-level metric
    'update_state' and mean aggregation. These issues are especially pertinent 
    where there are no damaged instances (recall error), or damaged predictions in  
    a batch (precision error). 
                     
        confusion:
        
                 true | 0   1  
            --------------------
            pred    0 | TN  FN
                    1 | FP  TP

        metrics:
        
            accuracy = (TN + TP) / (TN + FN + FP + TP)      
            precision = TP / (FP + TP)
            negative prediction value (npv) = TN / (FN + TN)
            recall = TP / (FN + TP)
            specificity = TN / (TN + FP)
            f1 = 2 * (precision * recall) / (precision + recall)

        undefined metrics: (in a batch) 
                                 acc pre npv rec spe f1
           IF only predict N          X               X   (FP = TP = 0)
           IF only predict P              X               (FN = TN = 0)
           IF only true N                     X       X   (FN = TP = 0)
           IF only true P                         X       (TN = FP = 0)

           calculations across an epoch ensure a sufficient distribution of 
           damaged and undamaged detections for calculation of confusion matrix metrics

    """

    CM_LOGS = ['tn', 'fn', 'fp', 'tp']

    def __init__(self, train_writer, val_writer, total_epochs, nlevels):
        super(EpochCM, self).__init__()
        self.train_writer = train_writer
        self.val_writer = val_writer
        self.total_epochs = total_epochs
        self.nlevels = nlevels

    # Initialise confusion matrix at epoch start

    def set_quant_attrs(self, isval = False):

        for quant in self.CM_LOGS: 
            if isval: 
                quant = "val_" + quant 
            setattr(self, quant, 0)
            for lvl in range(self.nlevels):
                name = quant + "_{:d}".format(lvl)
                setattr(self, name, 0)

    def on_epoch_begin(self, epoch, logs=None):        
        self.set_quant_attrs(False)
        
    def on_test_begin(self, logs=None):
        self.set_quant_attrs(True)
       
    # Update confusion matrix at batch end

    def update_quant_attr(self, logs, quant, isval=False, lvl=None):
      
        if lvl is not None: 
            quant += "_{:d}".format(lvl)            

        prop = "val_" + quant if isval else quant 
  
        current = getattr(self, prop)
        add = logs[quant]
        setattr(self, prop, current + add)

    def update_quant_attrs(self, logs, isval=False):

        for quant in self.CM_LOGS:            
            self.update_quant_attr(logs, quant, isval)
            for lvl in range(self.nlevels):
                self.update_quant_attr(logs, quant, isval, lvl)

    def on_train_batch_end(self, batch, logs=None):
        self.update_quant_attrs(logs, isval=False)

    def on_test_batch_end(self, batch, logs=None):
        self.update_quant_attrs(logs, isval=True)
    
    # Compute epoch confusion matrix metrics

    def process_attrs(self, writer, isval, lvl=None):

        prefix = "val_" if isval else ""
        suffix = "_{:d}".format(lvl) if lvl is not None else ""

        tn, fn, fp, tp = [getattr(self, prefix + quant + suffix) for quant in self.CM_LOGS]
        eps = 0.0000001

        acc = (tp + tn) / (tn + fn + fp + tp + eps) 
        pre = tp / (fp + tp + eps)
        npv = tn / (fn + tn + eps)
        rec = tp / (fn + tp + eps)
        spe = tn / (tn + fp + eps)
        f1 = 2 * (pre * rec) / (pre + rec + eps)

        cls_suffix = "  " if suffix == "" else suffix 
        cls_base = "{:s} cls".format("V" if isval else "T")

        print("{:s}   acc: {:.2f} | pre: {:.2f} | npv: {:.2f} | rec: {:.2f} | spe: {:.2f} | f1: {:.2f}"
              .format(cls_base + cls_suffix, acc, pre, npv, rec, spe, f1))

        with writer.as_default():
            tf.summary.scalar('cls_accuracy' + suffix, acc, step=self.total_epochs)
            tf.summary.scalar('cls_precision' + suffix, pre, step=self.total_epochs)
            tf.summary.scalar('cls_npv' + suffix, npv, step=self.total_epochs)
            tf.summary.scalar('cls_recall' + suffix, rec, step=self.total_epochs)
            tf.summary.scalar('cls_specificity' + suffix, spe, step=self.total_epochs)
            tf.summary.scalar('cls_f1' + suffix, f1, step=self.total_epochs)

        confusion = np.array([[tn, fn], 
                              [fp, tp]])

        PlotConfusionMatrix(writer, confusion, self.total_epochs, lvl)

    def on_epoch_end(self, epoch, logs=None):

        levels = [None] + list(range(self.nlevels))

        for isval in [False, True]:
            
            writer = self.train_writer if not isval else self.val_writer

            for lvl in levels:
                self.process_attrs(writer, isval, lvl)

        self.total_epochs += 1
            
    
    

   
                        


  
