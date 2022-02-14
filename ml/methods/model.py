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
import json

from absl import logging

import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as KL 
import tensorflow.keras.metrics as KM
import tensorflow.keras.regularizers as KR    
import tensorflow.keras.callbacks as KC
import tensorflow.keras.backend as KB

from methods._model.yolo import RpnGraph
from methods._model.head import ROIAlign, PoseGraph
from methods._data.targets import transform_targets
from methods._data import convert
from methods.loss import RpnLossLayer, PoseLossLayer, CombineLossesLayer
from methods.metrics import PlotConfusionMatrix, ClsMetricLayer, AllMetricLayer, ComputeAP, Compute3DIOU
from methods.visualise import get_ann_images

# CUSTOM_OBJECTS = {'YoloBox': }

# B: Collate model ===========================================================================

class DRSYolo():

    """Encapsulates DRSYolo functionality"""

    def __init__(self, mode, cfg, infer_ds=None, verbose=False):

        """
        :param mode: "training", "detection"
        :param cfg: configuration settings
        """

        assert mode in ['training', 'detection']
        self.mode = mode
        self.cfg = cfg
        self.infer_ds = infer_ds
        self.verbose = verbose

    def build_model(self):

        """Build DRSYolo architecture
        
        :param: see init

        :input_image:     [nbatch, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.CHANNELS]
        :input_rpn:       [nbatch, cfg.MAX_GT_INSTANCES, bbox [4] + cls [1]] 
        :input_pose:      [nbatch, cfg.MAX_GT_INSTANCES, center [2] + depth [1] + dims [3] + quart [4]]       
        """

        mode = self.mode
        cfg = self.cfg

        # > Inputs

        input_image = KL.Input([cfg['IMAGE_SIZE'], cfg['IMAGE_SIZE'], cfg['CHANNELS']], name='input_image')                

        if self.mode == 'training': 

            camera_fx = KL.Input([1], name='camera_fx')
            camera_fy = KL.Input([1], name='camera_fy')
            camera_h = KL.Input([1], name='camera_h')
            camera_R = KL.Input([3, 3], name='camera_R')
            camera_rot = KL.Input([3], name='camera_rot')            
            camera_w = KL.Input([1], name='camera_w')
                        
            input_environment = KL.Input([3], name = 'input_environment')
            input_features = KL.Input([cfg['MAX_GT_INSTANCES'], 3], name = 'input_features')
            input_pose = KL.Input([cfg['MAX_GT_INSTANCES'], 10], name='input_pose')   
            input_rpn = KL.Input([cfg['MAX_GT_INSTANCES'], 5], name='input_rpn')
            
            inputs = [camera_fx, camera_fy, camera_h, camera_R, camera_rot, camera_w, 
                        input_environment, input_features, input_image, input_pose, input_rpn]

            cameras = {'rot': camera_rot, 
                        'R': camera_R, 
                        'w': camera_w, 
                        'h': camera_h, 
                        'fx': camera_fx, 
                        'fy': camera_fy}

            gt_boxes = KL.Lambda(lambda x: transform_targets(*x, cfg=cfg), name='gt_boxes')([input_rpn, input_pose])

        # > RPN   

        rpn_fmaps, pd_boxes, pd_rpn = RpnGraph(input_image, mode, cfg)    

        if self.mode == "training":
            gt_rpn = input_rpn
            rpn_roi = gt_rpn[..., :4]                                                # [nbatch, cfg.MAX_GT_INSTANCES, (x1, y1, x2, y2)]
            gt_pose = tf.concat([rpn_roi, input_pose], axis = -1, name='gt_pose')    # [nbatch, cfg.MAX_GT_INSTANCES, 4 + 10]

        roi_align = ROIAlign((rpn_roi, rpn_fmaps), cfg)                                

        # > HEAD

        pd_pose = PoseGraph((rpn_roi, roi_align), cfg)                               # [nbatch, cfg.MAX_GT_INSTANCES, 10] (recollected)

        if self.mode == 'detection': 
            outs = tf.concat((pd_rpn, pd_pose), axis = -1)
            return K.Model(input_image, outs, name='drs_yolo') 

        # > LOSSES
        # note: only training mode left
        # note: ensure model json serialisable: pruning, quantisation

        rpn_loss = RpnLossLayer(cfg, name='rpn_loss')((pd_boxes, gt_boxes))        
        pose_loss = PoseLossLayer(cfg, name='pose_loss')((pd_pose, gt_pose)) 
        losses = [rpn_loss, pose_loss]

        # > METRICS
        
        if self.verbose:
            cls_metric = ClsMetricLayer(cfg, name='cm_metric')((pd_boxes, gt_boxes))
            all_metric = AllMetricLayer(cfg, name='all_metric')((pd_rpn, gt_rpn, input_features, pd_pose, input_pose, cameras)) 
                                                                 
            losses += [cls_metric, all_metric]

        loss = CombineLossesLayer(cfg, self.verbose, name='loss')(losses)        
        return K.Model(inputs, [loss], axis =-1)], name='drs_yolo')         # tf.concat([gt_rpn, pd_pose])

    def build(self, weight_path="weights"):

        """Build, compile, set layer trainability and load weights (if relevant)"""

        self.model = self.build_model()
        self.compile()

        if self.mode == 'training':
            self.freeze_darknet_layers()
            self.load_weights(optimiser=True)

        elif self.mode == 'detection':            
            self.model.trainable = False
            self.load_weights(optimiser=False, weight_path=weight_path)

    def load_weights(self, optimiser=True, weight_path="weights"):

        """Load saved model weights OR intialise default yolo backend 
        :note: weight transfer assumed better than weight initialisation strategies.
            COCO initialised YOLO already capable of identifying vehicle types

        :param weight_path: path to model checkpoints
                            default coco-trained yolo weights OR 
                            user-specified full-model weight file

        :result: model weights initialised
        """

        ckpt_dir = os.path.join(self.cfg['MODEL_DIR'], "checkpoints")        
        weight_path = os.path.join(ckpt_dir, weight_path) 

        if self.mode == "training":
            if os.path.isdir(ckpt_dir):                                            
                if optimiser:
                    optim_path = os.path.join(ckpt_dir, "optimiser.pkl") 
                    if os.path.isfile(optim_path):                  # important: set model weights after optimizer (otherwise reset)
                        with open(optim_path, 'rb') as f:
                            weights = pickle.load(f)
                        try:
                            self.model.optimizer.set_weights(weights)                          
                            logging.info('optimiser weights loaded from {:s}'.format(optim_path))
                        except:
                            logging.info('unable to load optimiser state')
                self.model.load_weights(weight_path, by_name=False) # by_name=False --> loading from tensorflow format topology            
                logging.info('model weights loaded: {:s}'.format(self.mode))    
                           
            else:
                convert.load_yolo_weights(self)
                logging.info('pretrained weights loaded: {:s}'.format(self.mode))
        else:
            train = DRSYolo("training", self.cfg)
            train.build()
            train.model.load_weights(weight_path, by_name=False)

            DRSYolo.transfer_weights(train.model, self.model)
            logging.info('model weights loaded: {:s}'.format(self.mode))       
    
    @staticmethod
    def transfer_weights(model_a, model_b):

        """Transfer model A weights to model B"""

        a_weight_names = [submod.name for submod in model_a.layers 
                            if len(submod.get_weights()) > 0]

        b_weight_names = [submod.name for submod in model_b.layers 
                            if len(submod.get_weights()) > 0]

        # quick fix: unordered names in detect vs train model

        lnames = []
        canames = a_weight_names.copy()
        cbnames = b_weight_names.copy()

        for aname, bname in zip(a_weight_names, b_weight_names):
            if aname[:4] == bname[:4]:
                break 
            else:
                bname_same = [bname for bname in cbnames if aname[:4] == bname[:4]]
                if "conv" not in aname:
                    bname = bname_same[0]
                else:       
                    bname_order = [int(bname[bname.rfind("_"):]) for bname in bname_same]
                    bname = bname_same[bname_order.index(min(bname_order))]

            lnames.append((aname, bname))
            canames.remove(aname)
            cbnames.remove(bname)
        
        for aname, bname in lnames:
            print(aname, bname)

        for aname, bname in lnames: # zip(a_weight_names, b_weight_names):

            weights = model_a.get_layer(aname).get_weights()     
            model_b.get_layer(bname).set_weights(weights)

    @staticmethod
    def restore(cfg, infer_ds, verbose=False):
        
        """Restores DRSYolo class during training

        :note: tensorflow bug in full model restore
               AttributeError: '_UserObject' object has no attribute 'summary'
               load model and optimiser weights (in the interim)
        """

        #full_model_dir = os.path.join(cfg['MODEL_DIR'], 'model')
        cls = DRSYolo("training", cfg, infer_ds, verbose)
        #cls.model = K.models.load(full_model_dir)
        return cls

    def freeze_darknet_layers(self):

        """Unfreeze darknet conv2d, batch-norm pairs assorted from back to front
        :note: darknet submodel initially frozen, then thawed

        :param model: full drs model
        :param no_unfreeze: number/percentage of darknet layers to thaw 
        :option isper: whether no_unfreeze is a percentage

        :result: modified darknet model
                 number of layers unfrozen
        """
        
        pfreeze = self.cfg['PER_DARKNET_FROZEN']

        # CASE 1: All layers trainable (default) 

        if pfreeze == 0:
            return

        # CASE 2: Some layers trainable

        trainable_layers = [layer for layer in self.model.layers if 
                            ('conv2d' in layer.name) or  
                            ('batch_norm' in layer.name)]
        ntrainable = len(trainable_layers)
        npairs = ntrainable // 2

        logging.info('layers: [pairs] {:d} [trainable] {:d} [total] {:d}'.format(npairs, ntrainable, len(self.model.layers)))        

        if 0 < pfreeze < 1:

            freeze_ind = 2 * round(pfreeze * npairs) 
            layers = trainable_layers[:freeze_ind]
            ntrainable = len(layers)

            for layer in layers:
                layer.trainable = False

        # CASE 3: All layers frozen

        else:
            ntrainable = 0
            for layer in trainable_layers:
                layer.trainable = False
        
        logging.info('layers frozen: {:d}'.format(ntrainable))

    def compile(self):

        """Compiles model object. Adds losses, metrics and regularisation prior 
        to compilation. Disables training of batch normalisation layers (if specified)
        """

        cfg = self.cfg     
        optimizer = K.optimizers.Adam(learning_rate=cfg['LR_INIT'], 
                                      clipnorm=cfg['GRADIENT_CLIPNORM'])
        
        # Add: Regularisation loss [L2]
        # note: skip gamma and beta weights of batch normalization layers.
        # ADJUST: division by size

        reg_losses = [KR.l2(cfg['WEIGHT_DECAY'])(w) / tf.cast(tf.size(w), tf.float32)
                        for w in self.model.trainable_weights
                        if 'gamma' not in w.name and 'beta' not in w.name]
        
        tot_reg_losses = tf.add_n(reg_losses, 'reg_losses')
        self.model.add_loss(lambda: tot_reg_losses)                    

        # Disable: Training of batch normalisation layers

        if not cfg['TRAIN_BN']:
            bn_layers = [l for l in self.model.layers if ('batch_normalization' in l.name)]
            for bn_layer in bn_layers:
                bn_layer.trainable = False
    
        self.model.compile(optimizer=optimizer,
                           loss=[None]*len(self.model.outputs),
                           run_eagerly=False)

    def get_callbacks(self):        
        
        train_dir = os.path.join(self.cfg['LOG_DIR'], 'cm_train')
        train_fwriter = tf.summary.create_file_writer(train_dir)
        
        val_dir = os.path.join(self.cfg['LOG_DIR'], 'cm_validation')
        val_fwriter = tf.summary.create_file_writer(val_dir)
        
        infer_dir = os.path.join(self.cfg['LOG_DIR'], 'infer')
        infer_writer = tf.summary.create_file_writer(infer_dir) 

        ckpt_dir = os.path.join(self.cfg['MODEL_DIR'], "checkpoints")    
   
        callbacks = [
                KC.ReduceLROnPlateau(patience=3, verbose=2, min_lr = self.cfg['LR_END']),
                KC.EarlyStopping(patience=5, verbose=2),            
                KC.TensorBoard(log_dir=self.cfg['LOG_DIR'], 
                               histogram_freq=5, 
                               write_images=True,
                               write_graph=True, 
                               update_freq='epoch'), 
                ModelRestartCheckpoint(ckpt_dir, self.cfg),
                ValImg(self.infer_ds, infer_writer, self.cfg)]

        if self.verbose:
                callbacks += [EpochCM(train_fwriter, 
                                      val_fwriter, 
                                      self.cfg['TOTAL_EPOCHS'], 
                                      len(self.cfg['MASKS'])),
                              MetricCB(train_fwriter, val_fwriter, self.cfg)]
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

    def predict(self, x):
        return self.model(x, training=False)

    def save(self, name = 'model', full=False, schematic=False):

        """Saves full model and optimiser weights (workaround)

        :note: optimiser weights saved separately | tensorflow
               error in model restore (revert when error resolved)
        """

        # Full model

        if full:
            full_model_dir = os.path.join(self.cfg['MODEL_DIR'], name)        
            self.model.save(full_model_dir, include_optimizer=False) 
            logging.info('model saved to {:s}'.format(full_model_dir))

        if schematic:

            """Plots model schematic to file"""

            plot_path = os.path.join(self.cfg['MODEL_DIR'], "model_schematic.png")

            self.model._layers = [layer for layer in self.model._layers if isinstance(layer, KL.Layer)]
            tf.keras.utils.plot_model(self.model, 
                                      to_file=plot_path, 
                                      show_layer_names=False, 
                                      expand_nested=True)
            
            logging.info("model schematic saved to {:s}".format(plot_path))       

    #def optimise(self, train_ds, val_ds):

        """
        prune: always first
        quant: always last
        """

        # Individual methods

    #    for mode in ['quant', 'wclust', 'prune']:
    #        opt_cls = Optimise(mode, self.copy(), train_ds, val_ds)
    #        opt_cls.fit_and_save(mode)

        # Combinations [prune > quant]

    #    prune = opt_cls        
    #    prune_quant = Optimise('quant', prune.copy(), train_ds, val_ds)
    #    prune_quant.fit_and_save('prune_quant')

        # All [prune > wclust > quant]
        
    #    prune_wclust = Optimise('wclust', prune, train_ds, val_ds)
    #    prune_wclust.fit_and_save('prune_wclust')

    #    prune_wclust_quant = Optimise('quant', prune_wclust, train_ds, val_ds)
    #    prune_wclust_quant.fit_and_save('prun_wclust_quant')
                
class ModelRestartCheckpoint(KC.Callback):    
    
    """Saves model weights of the best model, as defined by that that 
    yields the smallest 'val_loss' measure. Weights are saved to the
    checkpoint directory. The initial 'val_loss' is set to the 
    previously trained model's best 'val_loss'.
    """

    def __init__(self, ckpt_dir, cfg_mod):
        super(ModelRestartCheckpoint, self).__init__()

        self.current_ckpt_dir = os.path.join(ckpt_dir, 'weights')
        self.best_ckpt_dir = os.path.join(ckpt_dir, 'best_weights')

        init_val_loss = cfg_mod['BEST_VAL_LOSS']
        if init_val_loss == -1.0:
            init_val_loss = np.Inf
        self.init_val_loss = init_val_loss
        
        self.total_epochs = cfg_mod['TOTAL_EPOCHS']
        self.config_path = os.path.join(cfg_mod['MODEL_DIR'], "config.json")  
        self.cfg_mod = cfg_mod
    
    def on_train_begin(self, logs=None):
        self.best = self.init_val_loss

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_loss")
        self.model.save_weights(self.current_ckpt_dir)

        if np.less(current, self.best):
            logging.info('val loss improved from {:.0f} to {:.0f}. weights saved to {:s}'
                            .format(self.best, current, self.current_ckpt_dir))
            self.best = current
            self.model.save_weights(self.best_ckpt_dir)        
        else:
            logging.info('val loss did not improve.')

        self.total_epochs += 1
        self.save_cfg()
        self.save_optimiser()

    def save_cfg(self):

        """Save configuration attributes (excl. YOLO) to .json file
        
        :param ckpt_dir: general model saving directory
        :param name: model/test name
        
        :result: configuration attributes stored in config.json
        """

        cfg = self.cfg_mod.copy()

        cfg['TOTAL_EPOCHS'] = self.total_epochs
        cfg['BEST_VAL_LOSS'] = self.best if not np.isposinf(self.best) else -1.0

        with open(self.config_path, 'w') as f:
            json.dump(cfg, f, sort_keys=False, indent=4)  
            logging.info('updated cfg after {:d} epochs (total) to {:s}'.format(self.total_epochs, 
                                                                                self.config_path))

    def save_optimiser(self):

        optim_path = os.path.join(self.cfg_mod['MODEL_DIR'], 'checkpoints', 'optimiser.pkl')        
        weights = self.model.optimizer.get_weights()  # tf.keras.optimizers.get('Adam').get_weights()

        with open(optim_path, 'wb') as f:
            pickle.dump(weights, f)

        logging.info('optimiser weights saved to {:s}'.format(optim_path))

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

class ValImg(KC.Callback):

    """Plot a sample of validation images':
        > predicted and ground-truth image annotations
        > image gradients 
        > layer output filters
        to tensorboard. 
    Showcase model improvement across epoch intervals. 
    """

    LAYERS = ['conv2d', 'conv2d_1', 
              'darknet', 
              'yolo_conv_0', 'yolo_conv_1', 'yolo_conv_2', 
              'fpn_p0', 'fpn_p1', 'fpn_p2']

    GRAD_RPN_LAYERS = ['yolo_out_0', 'yolo_out_1', 'yolo_out_2']

    def __init__(self, infer_ds, infer_writer, cfg):
        super(ValImg, self).__init__()

        # recollect original images and annotations
       
        ds = list(infer_ds)[0][0]

        self.cameras = {'rot': ds['camera_rot'],
                        'R':   ds['camera_R'],
                        'w':   ds['camera_w'],
                        'h':   ds['camera_h'],
                        'fx':  ds['camera_fx'],
                        'fy':  ds['camera_fy']}
        self.images = ds['input_image']
        self.sample = tf.expand_dims(self.images[0], 0)
        self.nimage = len(self.images)
        self.cfg = cfg

        gt_padded_anns = tf.concat([ds['input_rpn'],   # [bbox, cls][5]
                                    ds['input_pose']],  # [center, depth, dims, quart][10]
                                    axis = -1)

        self.gt_ann_images = get_ann_images(gt_padded_anns, 
                                            self.cameras, 
                                            self.images, 
                                            self.cfg,
                                            size = self.cfg['IMAGE_SIZE'])

        self.infer_writer = infer_writer
        self.total_epochs = self.cfg['TOTAL_EPOCHS']

        # Create detection model
        
        cls = DRSYolo(mode = 'detection', cfg = self.cfg)
        cls.build()
        self.detect_model = cls.model

    def on_epoch_end(self, epoch, logs=None): 

        # if self.total_epochs > 1 & self.total_epochs % 5 == 0:  

        DRSYolo.transfer_weights(self.model, self.detect_model)
        self.plot_model_predictions()
        #self.plot_model_layers()

        self.total_epochs += 1

    def plot_model_predictions(self):
        
        pd_padded_anns = self.detect_model(self.images)
        pd_ann_images = get_ann_images(pd_padded_anns, 
                                       self.cameras, 
                                       tf.identity(self.gt_ann_images), 
                                       self.cfg,
                                       size = self.cfg['IMAGE_SIZE'])
            
        with self.infer_writer.as_default():
            tf.summary.image('Sample: Validation Data', 
                            pd_ann_images, 
                            max_outputs=self.nimage, 
                            step=self.total_epochs)

    #def plot_model_layers(self):
    #
    #    """Visualises 
    #        a. Initial convolutions
    #        b. Darknet output 
    #        c. Yolo feature maps 
    #        on a sample image"""

    #    """Extracts RPN and Head submodels from the detection model.
    #    This isolates the gradient features relating to each submodel.

    #        a. RPN
    #            input --> yolo_out_0
    #            input --> yolo_out_1
    #            input --> yolo_out_2 (if exists)

    #        b. Head
    #            roi_align --> pd_pose
             
    #    :param detect_model: detection model

    #    :result mod_rpn: rpn sub model
    #    """


        # A: Visualise activations and RPN gradients

    #    for layer in self.LAYERS:

    #        if layer not in self.model.layers:
    #            continue

            # 1: Create intermediary models 

    #        inter_model = K.Model(inputs=self.model.input,
    #                              outputs=self.model.get_layer(layer).output)
                       
    #        acts = inter_model.predict(self.sample)

            # 2: Generate overlaid activation heatmaps

    #        nacts = len(acts)

    #        if nouts == 1:
    #            acts = [acts]

    #        for i, act in enumerate(acts):
    #            fig = self.create_overlay(act)     
    #            hmap_name = f"heat map - {layer} - {i}"

    #            with self.infer_fwriter.as_default():                
    #                tf.summary.image(hmap_name, 
    #                                 fig, 
    #                                 max_outputs = nacts,
    #                                 step = self.total_epochs)

            # 3: Generate overlaid rpn gradient heatmaps (optional)

    #        if layer in self.GRAD_RPN_LAYERS:                 
    #            self.write_grad_summary(self.sample, acts[0], layer)

        # B: Visualise head layer gradients

    #    base_model = K.Model(inputs=self.model.input,
    #                         outputs=self.model.get_layer('roi_align').output)

    #    input = base_model.predict(self.sample)

    #    head_model = K.Model(inputs=self.model.get_layer('roi_align').input, 
    #                         outputs=self.model.get_layer('pd_pose').output)

    #    output = head_model.predict(input)
    #    self.write_grad_summary(input, output, 'pd_pose')
            
    #def create_heat_overlay(self, acts):

    #    """Modified https://github.com/philipperemy/keract/blob/master/keract/keract.py"""

        # 1: Homogenise image format

    #    input_image = tf.squeeze(self.sample, axis=0)
        
    #    nrow = int(acts.shape[-1]**0.5 - 0.001) + 1
    #    ncol = int(np.ceil(acts.shape[-1]) / nrow)

    #    fig, axes = plt.subplots(nrow, ncol, figsize = (12, 12))
     
    #    scaler = MinMaxScaler()
    #    scaler.fit(acts.reshape(-1, 1))

    #    isize = np.minimum(nrow * ncol, nchannel)
    #    irange = np.random.choice(np.range(nchannel), isize)
        
    #    for i, c in enumerate(irange):
    #        img = acts[0, :, :, c]
    #        img = scaler.transform(img)
    #        img = Image.fromarray(img)
    #        img = img.resize((input_image.shape[1], input_image.shape[0]), Image.LANCZOS)
    #        img = np.array(img)

    #        axes.flat[i].imshow(input_image / 255.0)
            # overlay the activation at 70% transparency onto the image with a heatmap colour scheme
            # Lowest activations are dark, highest are dark red, mid are yellow
    #        axes.flat[i].imshow(img, alpha=0.3, cmap='jet', interpolation='bilinear')
    #        axes.flat[i].axis('off') 

    #    return FigureWriter.encode_fig(fig)
       
    #def create_grad_overlay(self, grads):

    #    input_image = tf.squeeze(self.sample, axis=0)

    #    nrow = grads.shape[-1]
    #    ncol = grads.shape[-2]

    #    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12))

    #    scaler = MinMaxScaler()
    #    scaler.fit(grads.reshape(-1, 1))

    #    for i in range(nrow):
    #        for j in range(ncol):
    #            img = grads[:, :, j, i]
    #            img = scaler.transform(img)
    #            img = Image.fromarray(img)
    #            img = img.resize((input_image.shape[1], input_image.shape[0]), Image.LANCZOS)
    #            img = np.array(img)

    #            axes[i, j].imshow(input_image / 255.0)
                # overlay the activation at 70% transparency  onto the image with a heatmap colour scheme
                # Lowest activations are dark, highest are dark red, mid are yellow
    #            axes[i, j].imshow(img, alpha=0.3, cmap='jet', interpolation='bilinear')        
    #            axes[i, j].axis('off') 

    #    return FigureWriter.encode_fig(fig)

    #def write_grad_summary(self, input, output, layer):

        """Extract gradients, create image overlays and write output to tensorboard
        
        :param input: model input
        :param output: model output
        :param layer: output layer name
        """

    #    with tf.GradientTape() as t:                   
    #        grads = t.gradient(output, input)               
                
    #        fig = self.create_grad_overlay(grads)
    #        grad_name = f"grads - {layer}"

    #        with self.infer_fwriter.as_default():                
    #            tf.summary.image(grad_name, 
    #                             fig, 
    #                             step = self.total_epochs)

class MetricCB(KC.Callback):

    TYPES = ["all"] + list(ComputeAP.TYPES.keys())
    SETS = ["train", "val"]
    MEASURES = ["ap", "iou"]
    COLS = {'ap': ComputeAP.IOU_THRESHES,
            'iou': ['mask', 'lens']}

    def __init__(self, train_writer, val_writer, cfg):
        super(MetricCB, self).__init__()
        
        self.train_writer = train_writer
        self.val_writer = val_writer
        self.total_epochs = cfg['TOTAL_EPOCHS']
        self.cfg = cfg

        self.NMEASURES = len(self.MEASURES)
        self.NSETS = len(self.SETS)
        self.NTYPES = len(self.TYPES)

    def compute_and_update_ap(self, batch, isval=False):

        """Extracts layer output with ground-truth metric ids 
        and updates AP & 3D IoU"""

        attr_set = 'val' if isval else 'train'

        all_metric_layer = self.model.get_layer('all_metric')

        pd_rpn = all_metric_layer.pd_rpn
        gt_rpn = all_metric_layer.gt_rpn
        
        pd_pose = all_metric_layer.pd_pose
        gt_pose = all_metric_layer.gt_pose
        pose_ids = all_metric_layer.pose_ids

        cameras = {'rot': all_metric_layer.camera_rot, 
                   'R': all_metric_layer.camera_R, 
                   'w': all_metric_layer.camera_w, 
                   'h': all_metric_layer.camera_h,
                   'fx': all_metric_layer.camera_fx,
                   'fy': all_metric_layer.camera_fy}

        # Compute: AP and 3D IoU

        results = {'ap': ComputeAP(pd_rpn, gt_rpn).result,                                    # [ntype, niou] 
                   'iou': Compute3DIOU(cameras, pd_pose, gt_pose, pose_ids, self.cfg).result, # [ntype, [mask, lens]]
                   'nrpn': self.count_n_rpn(pd_rpn)}

        # Update model metrics       

        for m, measure in enumerate(self.MEASURES):
            result = results[measure]
            for t, type in enumerate(self.TYPES):                                
                attr_name_type = f"{measure}_{attr_set}_{type}"                
                old = getattr(self, attr_name_type)
                update = result[t]
                if tf.reduce_mean(update) != -1:
                    obatch = self.batch[m, int(isval), t]
                    new = old + 1/obatch * (update - old)             
                    setattr(self, attr_name_type, new)
                    self.batch = tf.tensor_scatter_nd_add(self.batch, [[m, int(isval), t]], [1])     
    
    def count_n_rpn(self, pd_rpn):

        """Counts the number of region proposals per batch.
        Tracks the slow initialisation/formulation of RPN
        relative to pose estimation.

        :param pd_rpn: [NBATCH, cfg.MAX_GT_INSTANCES, 7] 

        :result: number of region proposals per batch [1]
        """

        return tf.reduce_sum(tf.cast(tf.where(pd_rpn[..., 2] > 0), tf.int32)).numpy()

    def on_epoch_begin(self, epoch, logs=None):
    
        """Set all attribute ious to zero"""
    
        self.batch = tf.ones((self.NMEASURES, 
                              self.NSETS, 
                              self.NTYPES))
    
        for measure in self.MEASURES:
            zero = tf.zeros(len(self.COLS[measure]))
            for attr_set in self.SETS:
                for t, type in enumerate(self.TYPES):
                    attr_name_type = f"{measure}_{attr_set}_{type}"               
                    setattr(self, attr_name_type, zero)

    def on_train_batch_end(self, batch, logs=None): 
        self.compute_and_update_ap(batch, False)

    def on_test_batch_end(self, batch, logs=None):
        self.compute_and_update_ap(batch, True)
        
    def on_epoch_end(self, epoch, logs=None):

        for measure in self.MEASURES:            
            cols = self.COLS[measure]

            for set in self.SETS:
                writer = self.val_writer if set == 'val' else self.train_writer
                
                for type in self.TYPES:            
                    attr_name = f"{measure}_{set}_{type}"
                    ious = getattr(self, attr_name)
                    
                    run_log = "{:s} {:s}".format(measure.upper(), 
                                                 set[0].upper())
                    name_main = f"{measure}_{type}"

                    for c, col in enumerate(cols): 
                        
                        col_str = "{:.2f}".format(col) if not isinstance(col, str) else col
                        name = f"{name_main}_{col_str}"
                        iou = ious[c]

                        with writer.as_default():
                            tf.summary.scalar(name, iou, step = self.total_epochs)
            
                        run_log += " ({:s}) {:.2f}".format(col_str, iou)                                                         
                         
                    logging.info(run_log)

            self.total_epochs += 1

"""
    Description: Model optimisation
    Functions: TFlite, Pruning, Quantisation, Weight Clustering
"""

import tensorflow_model_optimization as tfmot
import zipfile
import tempfile

def export_tflite(model_path):

    """Exports saved model to .tflite
    
    :note: extensions (tensorflow help)
    :note: python implementation under active development (android / ios)
    :note: need 'CombinedNonMaxSuppression' added to selected tf flex delegates list

    :param model_path: SavedModel directory

    :result: saved tflite model
    """

    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    #converter.optimizations = [tf.lite.Optimize.DEFAULT] post-quantisation
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]
    
    tflite_model = converter.convert()

    tflite_path = model_path + '.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    logging.info('tflite model saved to {:s}'.format(tflite_path))


# PART B-D: Pruning / Quantisation / Weight CLustering ===========================

#class Optimise(DRSYolo):

    # Do: customise layer (remove bias) 
    #     save: config, weights
    #     restore

#    def __init__(self, mode, pretrain, train_ds, val_ds):

#        """
#        :note: restarts pruned model tensorboard
#
#        :param mode: prune, quant or wclust
#        :param train: DRSYolo object
#        """

#        self.mode = mode
#        self.model = pretrain.model
#        self.cfg = pretrain.cfg

#        self.train_ds = train_ds
#        self.val_ds = val_ds

    # Fit model

#    def build_model(self):

#        self.model = K.models.clone_model(
#            self.model, 
#            clone_function=self.apply_layers)

#        if self.mode == 'quant':
#            self.model = tfmot.quantization.keras.quantize_apply(self.model)


#    def fit_and_save(self, name):
#        self.build_model()
#        self.model.summary()

#        self.compile()

#        self.model.fit(self.train_ds, 
#                       self.val_ds, 
#                       epochs=cfg['OPTIMISE_EPOCHS'],
#                       callbacks=self.get_callbacks(),
#                       verbose=2)

#        self.save(name)
    
    # Callbacks

#    def get_callbacks(self):

#        if self.mode == 'prune':
#            log_dir = os.path.join(self.cfg['LOG_DIR'], "prune")

#            update = tfmot.sparsity.keras.UpdatePruningStep()
#            summaries = tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)
                
#            return [update, summaries]

#        return None
    
    # Apply optimsation

#    def apply_layers(self, layer):        
#        if layer.name in cfg['OPTIMISE_LAYERS']:
#            if self.mode == 'prune':
#                return tfmot.sparsity.keras.prune_low_magnitude(layer)
#            if self.mode == 'quant':
#                return tfmot.quantization.keras.qunatize_annotate_layer(layer)
#            if self.mode == 'wclust':
#                return tfmot.clustering.keras.cluster_weights(layer, self.cfg['CLUSTER_PARAMS'])
#        return layer

    # Get model size

#    def strip_model(self):
#        if self.mode == 'prune':
#            self.model = tfmot.sparsity.keras.strip_pruning(self.model)
#        if self.mode == 'wclust':
#            self.model = tfmot.clustering.keras.strip_clustering(self.model)    

#    def save_model_file(self):
#        _, keras_file = tempfile.mkstemp('.h5')
#        self.strip_model()
#        self.model.save(keras_file, include_optimizer=False)
#        return keras_file

#    def get_gzipped_model_size(self):

#        """
#        :return: gzipped model size in bytes
#        """

#        keras_file = self.save_model_file()

#        _, zipped_file = tempfile.mkstemp('.zip')
#        with zipfile.ZipFile(zipped_file, 
#                             'w', 
#                             compression=zipfile.ZIP_DEFLATED) as f:
#            f.write(keras_file)
#            return os.path.getsize(zipped_file)