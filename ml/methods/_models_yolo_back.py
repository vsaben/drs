import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2

def DarknetConv(x, filters, size, strides=1, batch_norm=True, activate_type = 'leaky'):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        if activate_type == 'leaky': x = LeakyReLU(alpha=0.1)(x)
        if activate_type == 'mish': x = mish(x)
    return x

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

def DarknetResidual(x, filters, activate_type = 'leaky'): 
    prev = x
    x = DarknetConv(x, filters // 2, 1, activate_type = activate_type)
    x = DarknetConv(x, filters, 3, activate_type = activate_type)
    x = Add()([prev, x])
    return x

def CSPDarknetResidual(x, filters, activate_type = 'mish'):
    prev = x
    x = DarknetConv(x, filters, 1, activate_type = activate_type)
    x = DarknetConv(x, filters, 3, activate_type = activate_type)
    x = Add()([prev, x])
    return x

def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x

def CSPDarknetBlock(x, filters, blocks, residual_type = 'yolo4'):
    rt = DarknetConv(x, filters, 1, activate_type = 'mish')
    x = DarknetConv(x, filters, 1, activate_type= 'mish')

    DarknetResidualFunc = CSPDarknetResidual if residual_type == 'yolo4' else DarknetResidual
    for _ in range(blocks):
        x = DarknetResidualFunc(x, filters, activate_type = 'mish')
    x = DarknetConv(x, filters, 1, activate_type= 'mish')
    
    x = Add()([x, rt])
    return x

def Darknet(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  # skip connection
    x = x_36 = DarknetBlock(x, 256, 8)  # skip connection
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


def CSPDarknet(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 32, 3, activate_type = 'mish')
    x = DarknetConv(x, 64, 3, 2, activate_type = 'mish')    
    x = CSPDarknetBlock(x, 64, 1, residual_type = 'yolo3')
    x = DarknetConv(x, 64, 1, activate_type = 'mish')
    x = DarknetConv(x, 128, 3, 2, activate_type = 'mish') 
    x = CSPDarknetBlock(x, 64, 2)
    x = DarknetConv(x, 128, 1, activate_type = 'mish')
    x = DarknetConv(x, 256, 3, 2, activate_type = 'mish') 
    x = CSPDarknetBlock(x, 128, 8)
    x = x_1 = DarknetConv(x, 256, 1, activate_type = 'mish')
    x = DarknetConv(x, 512, 3, 2, activate_type = 'mish')
    x = CSPDarknetBlock(x, 256, 8)
    x = x_2 = DarknetConv(x, 512, 1, activate_type = 'mish')
    x = DarknetConv(x, 1024, 3, 2, activate_type = 'mish')
    x = CSPDarknetBlock(x, 512, 4)
    
    x = DarknetConv(x, 1024, 1, activate_type = 'mish')
    x = DarknetConv(x, 512, 1)
    x = DarknetConv(x, 1024, 3)
    x = DarknetConv(x, 512, 1)

    x = Add()([CSPMaxPool(13)(x), CSPMaxPool(9)(x), CSPMaxPool(5)(x), x])
    x = DarknetConv(x, 512, 1)
    x = DarknetConv(x, 1024, 3)
    x = DarknetConv(x, 512, 1)
    return tf.keras.Model(inputs, (x_1, x_2, x), name=name)

def CSPMaxPool(ksize):
    return Lambda(lambda x: tf.nn.max_pool(x, ksize, strides=1, padding='SAME'))

def DarknetTiny(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 16, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 32, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 64, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 128, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = x_8 = DarknetConv(x, 256, 3)  # skip connection
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 512, 3)
    x = MaxPool2D(2, 1, 'same')(x)
    x = DarknetConv(x, 1024, 3)
    return tf.keras.Model(inputs, (x_8, x), name=name)

def CSPDarknetBlockTiny(x, filters, return_rt_last = False):
    rt0 = x
    x = route_group(x, 2, 1)
    x = rt1 = DarknetConv(x, filters, 3)
    x = DarknetConv(x, filters, 3)
    x = Add()([x, rt1])
    x = rt_last = DarknetConv(x, 2*filters, 1)
    x = Add()([rt0, x])
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 4*filters, 3)
    if return_rt_last: return x, rt_last
    return x

def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]

def CSPDarknetTiny(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 32, 3, 2)
    x = DarknetConv(x, 64, 3, 2)
    x = DarknetConv(x, 64, 3)
    x = CSPDarknetBlockTiny(x, 32)
    x = CSPDarknetBlockTiny(x, 64)
    x, rt1 = CSPDarknetBlockTiny(x, 128, return_rt_last = True)
    return tf.keras.Model(inputs, (rt1, x), name=name)