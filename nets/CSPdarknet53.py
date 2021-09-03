from functools import wraps
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, Concatenate, Layer
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from utils.utils import compose


class Mish(Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


# --------------------------------------------------#
#   A single convolution
# --------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4),
                           'padding': 'valid' if kwargs.get('strides') == (2, 2) else 'same'}
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


# ---------------------------------------------------#
#   
#   DarknetConv2D + BatchNormalization + Mish
# ---------------------------------------------------#
def DarknetConv2D_BN_Mish(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Mish())


# ---------------------------------------------------#
#   Building blocks for CSPdarknet
#   There's a big residual edge
#   This big residual edge bypasses a lot of residual structures
# ---------------------------------------------------#
def resblock_body(x, num_filters, num_blocks, all_narrow=True):
    # Compression of length and width
    preconv1 = ZeroPadding2D(((1, 0), (1, 0)))(x)
    preconv1 = DarknetConv2D_BN_Mish(num_filters, (3, 3), strides=(2, 2))(preconv1)

    # It makes a big residual edge
    shortconv = DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(preconv1)

    # The convolution of the trunk
    mainconv = DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(preconv1)
    # 1x1 convolution integrates the number of channels - > Feature extraction using 3x3 convolution and residual structure
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Mish(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (3, 3)))(mainconv)
        mainconv = Add()([mainconv, y])
    # 1x1 convolved with the residual edge
    postconv = DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(mainconv)
    route = Concatenate()([postconv, shortconv])

    # Finally, the number of channels is integrated
    return DarknetConv2D_BN_Mish(num_filters, (1, 1))(route)


# ---------------------------------------------------#
#   darknet53 The main part of darknet53
# ---------------------------------------------------#
def darknet_body(x):
    x = DarknetConv2D_BN_Mish(32, (3, 3))(x)
    x = resblock_body(x, 64, 1, False)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    feat1 = x
    x = resblock_body(x, 512, 8)
    feat2 = x
    x = resblock_body(x, 1024, 4)
    feat3 = x
    return feat1, feat2, feat3
