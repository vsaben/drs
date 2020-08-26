# Description: Collect YOLO-base functions and perform base model selection
# Function:
#   A. Choose base YOLO model and associated cfg
#   B. Base YOLO models
#      B1. YoloV3
#      B2. YoloV3Tiny
#      B3. YoloV4
#      B4. YoloV4Tiny
#   C. Freeze specified layers

# A: Choose base YOLO model and associated cfg ==============================

def choose_base_model(isyolov4, istiny, cfg):
    if (not isyolov4) and (not istiny): return YoloV3, cfg.YOLO.V3
    if (not isyolov4) and istiny: return YoloV3Tiny, cfg.YOLO.V3T
    if isyolov4 and (not istiny): return YoloV4, cfg.YOLO.V4
    if isyolov4 and istiny: return YoloV4Tiny, cfg.YOLO.V4T

# B: Base YOLO model ========================================================

from tensorflow.keras import Model
from tensorflow.keras.layers import Lambda, Input, UpSampling2D, Concatenate
from methods import _models_yolo_back as yolo_back
from methods import _models_yolo_head as yolo_head

from config import cfg

def YoloV3(cfg_mod = cfg.YOLO.V3, size=None, channels=3, classes=1, training=False):

    # Adaptation: Change class probability for single value indicating the probability of damage

    anchors = cfg_mod.ANCHORS.numpy()
    masks = cfg_mod.MASKS.numpy().tolist()

    x = inputs = Input([size, size, channels], name='input')
    x_36, x_61, x = yolo_back.Darknet(name='yolo_darknet')(x)

    x = yolo_head.YoloConv(512, name='yolo_conv_0')(x)
    out_0 = yolo_head.YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

    x = yolo_head.YoloConv(256, name='yolo_conv_1')((x, x_61))
    out_1 = yolo_head.YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

    x = yolo_head.YoloConv(128, name='yolo_conv_2')((x, x_36))
    out_2 = yolo_head.YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

    if training: return Model(inputs, (out_0, out_1, out_2), name='yolov3')

    boxes_0 = Lambda(lambda x: yolo_head.Boxes(x, anchors[masks[0]], classes), name='yolo_boxes_0')(out_0)
    boxes_1 = Lambda(lambda x: yolo_head.Boxes(x, anchors[masks[1]], classes), name='yolo_boxes_1')(out_1)
    boxes_2 = Lambda(lambda x: yolo_head.Boxes(x, anchors[masks[2]], classes), name='yolo_boxes_2')(out_2)
    outputs = Lambda(lambda x: yolo_head.NMS(x, anchors, masks, classes), name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov3')


def YoloV3Tiny(cfg_mod = cfg.YOLO.V3T, size=None, channels=3, classes=1, training=False):

    anchors = cfg_mod.ANCHORS.numpy()
    masks = cfg_mod.MASKS.numpy().tolist()

    x = inputs = Input([size, size, channels], name='input')
    x_8, x = DarknetTiny(name='yolo_darknet')(x)

    x = yolo_head.YoloConvTiny(256, name='yolo_conv_0')(x)
    out_0 = yolo_head.YoloOutput(256, len(masks[0]), classes, name='yolo_output_0')(x)

    x = yolo_head.YoloConvTiny(128, name='yolo_conv_1')((x, x_8))
    out_1 = yolo_head.YoloOutput(128, len(masks[1]), classes, name='yolo_output_1')(x)

    if training: return Model(inputs, (out_0, out_1), name='yolov3')

    boxes_0 = Lambda(lambda x: yolo_head.Boxes(x, anchors[masks[0]], classes), name='yolo_boxes_0')(out_0)
    boxes_1 = Lambda(lambda x: yolo_head.Boxes(x, anchors[masks[1]], classes), name='yolo_boxes_1')(out_1)
    outputs = Lambda(lambda x: yolo_head.NMS(x, anchors, masks, classes), name='yolo_nms')((boxes_0[:3], boxes_1[:3]))
    
    return Model(inputs, outputs, name='yolov3_tiny')

def YoloV4(cfg_mod = cfg.YOLO.V4, size=None, channels=3, classes=1, training=False):
    
    anchors = cfg_mod.ANCHORS.numpy()
    masks = cfg_mod.MASKS.numpy().tolist()

    x = inputs = Input([size, size, channels], name='input')
    rt1, rt2, rt0 = yolo_back.CSPDarknet(name='yolo_darknet')(x)

    x = rt0
    x = rt2 = yolo_head.YoloConv(256, name='yolo_pre_conv_1', iscsp = True)((x, rt2))
    x = rt1 = yolo_head.YoloConv(128, name='yolo_pre_conv_a', iscsp = True)((x, rt1))
    
    out_2 = yolo_head.YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)
    
    x = yolo_back.DarknetConv(x, 256, 3, 2)
    x = Concatenate()([x, rt2])
    x = yolo_head.YoloConv(256, name='yolo_conv_1')(x)

    out_1 = yolo_head.YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

    x = yolo_back.DarknetConv(x, 512, 3, 2)
    x = Concatenate()([x, rt0])
    x = yolo_head.YoloConv(512, name='yolo_conv_0')(x)

    out_0 = yolo_head.YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

    if training: return Model(inputs, (out_0, out_1, out_2), name='yolov3')

    boxes_0 = Lambda(lambda x: yolo_head.Boxes(x, anchors[masks[0]], classes), name='yolo_boxes_0')(out_0)
    boxes_1 = Lambda(lambda x: yolo_head.Boxes(x, anchors[masks[1]], classes), name='yolo_boxes_1')(out_1)
    boxes_2 = Lambda(lambda x: yolo_head.Boxes(x, anchors[masks[2]], classes), name='yolo_boxes_2')(out_2)
    outputs = Lambda(lambda x: yolo_head.NMS(x, anchors, masks, classes), name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov4')
    

def YoloV4Tiny(cfg_mod = cfg.YOLO.V4T, size=None, channels=3, classes=1, training=False):
    
    anchors = cfg_mod.ANCHORS.numpy()
    masks = cfg_mod.MASKS.numpy().tolist()

    x = inputs = Input([size, size, channels], name='input')
    rt1, x = yolo_back.CSPDarknetTiny(name='yolo_darknet')(x)

    x = yolo_back.DarknetConv(x, 256, 1)
    out_0 = yolo_head.YoloOutput(256, len(masks[0]), classes, name='yolo_output_0')(x)

    x = yolo_back.DarknetConv(x, 128, 1)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, rt1])

    out_1 = yolo_head.YoloOutput(128, len(masks[1]), classes, name='yolo_output_1')(x)

    if training: return Model(inputs, (out_0, out_1), name='yolov4')

    boxes_0 = Lambda(lambda x: yolo_head.Boxes(x, anchors[masks[0]], classes), name='yolo_boxes_0')(out_0)
    boxes_1 = Lambda(lambda x: yolo_head.Boxes(x, anchors[masks[1]], classes), name='yolo_boxes_1')(out_1)
    outputs = Lambda(lambda x: yolo_head.NMS(x, anchors, masks, classes), name='yolo_nms')((boxes_0[:3], boxes_1[:3]))
    
    return Model(inputs, outputs, name='yolov4_tiny')
 

# C: Freeze specified layers ==================================================================

def freeze_layers(model, setting):
    pass
