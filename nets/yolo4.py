from functools import wraps
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D, SeparableConv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.applications.mobilenetv2 import MobileNetV2
from keras.regularizers import l2
from utils.utils import compose


# --------------------------------------------------#
#   A single convolution
# --------------------------------------------------#
@wraps(Conv2D)
@wraps(SeparableConv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4),
                           'padding': 'valid' if kwargs.get('strides') == (2, 2) else 'same'}
    darknet_conv_kwargs.update(kwargs)
    return SeparableConv2D(*args, **darknet_conv_kwargs)


# ---------------------------------------------------#
#   
#   DarknetConv2D + BatchNormalization + LeakyReLU
# ---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


# ---------------------------------------------------#
#   Characteristics of layer
# ---------------------------------------------------#
def make_five_convs(x, num_filters):
    # Five times convolution
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    return x


# ---------------------------------------------------#
#   Characteristics of layer
# ---------------------------------------------------#
def yolo_body(inputs, num_anchors, num_classes):
    # Build the trunk model for darknet53
    mobilenetv2 = MobileNetV2(input_tensor=inputs, weights=None)
    feat3 = mobilenetv2.get_layer('block_16_project_BN').output
    feat2 = mobilenetv2.get_layer('block_12_project_BN').output
    feat1 = mobilenetv2.get_layer('block_5_project_BN').output
    # feat1,feat2,feat3 = darknet_body(inputs)

    # The first feature layer
    # y1=(batch_size,13,13,3,85)
    p5 = DarknetConv2D_BN_Leaky(512, (1, 1))(feat3)
    p5 = DarknetConv2D_BN_Leaky(1024, (3, 3))(p5)
    p5 = DarknetConv2D_BN_Leaky(512, (1, 1))(p5)
    # Use the SPP structure
    maxpool1 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(p5)
    maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(p5)
    maxpool3 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(p5)
    p5 = Concatenate()([maxpool1, maxpool2, maxpool3, p5])
    p5 = DarknetConv2D_BN_Leaky(512, (1, 1))(p5)
    p5 = DarknetConv2D_BN_Leaky(1024, (3, 3))(p5)
    p5 = DarknetConv2D_BN_Leaky(512, (1, 1))(p5)

    p5_upsample = compose(DarknetConv2D_BN_Leaky(256, (1, 1)), UpSampling2D(2))(p5)

    p4 = DarknetConv2D_BN_Leaky(256, (1, 1))(feat2)
    p4 = Concatenate()([p4, p5_upsample])
    p4 = make_five_convs(p4, 256)

    p4_upsample = compose(DarknetConv2D_BN_Leaky(128, (1, 1)), UpSampling2D(2))(p4)

    p3 = DarknetConv2D_BN_Leaky(128, (1, 1))(feat1)
    p3 = Concatenate()([p3, p4_upsample])
    p3 = make_five_convs(p3, 128)

    p3_output = DarknetConv2D_BN_Leaky(256, (3, 3))(p3)
    p3_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(p3_output)

    # 26,26 output
    p3_downsample = ZeroPadding2D(((1, 0), (1, 0)))(p3)
    p3_downsample = DarknetConv2D_BN_Leaky(256, (3, 3), strides=(2, 2))(p3_downsample)
    p4 = Concatenate()([p3_downsample, p4])
    p4 = make_five_convs(p4, 256)

    p4_output = DarknetConv2D_BN_Leaky(512, (3, 3))(p4)
    p4_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(p4_output)

    # 13,13 output
    p4_downsample = ZeroPadding2D(((1, 0), (1, 0)))(p4)
    p4_downsample = DarknetConv2D_BN_Leaky(512, (3, 3), strides=(2, 2))(p4_downsample)
    p5 = Concatenate()([p4_downsample, p5])
    p5 = make_five_convs(p5, 512)

    p5_output = DarknetConv2D_BN_Leaky(1024, (3, 3))(p5)
    p5_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(p5_output)

    return Model(inputs=inputs, outputs=[p5_output, p4_output, p3_output])


# ---------------------------------------------------#
#   Adjust each characteristic layer of the predicted value to the true value
# ---------------------------------------------------#
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    # [1, 1, 1, num_anchors, 2]
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    # Get the grid of x and y
    # (13,13, 1, 2)
    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    # (batch_size,13,13,3,85)
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust the predicted value to the real value
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    # The following parameters are returned when calculating loss
    if calc_loss is True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


# ---------------------------------------------------#
#   Adjust the box so that it looks like the real picture
# ---------------------------------------------------#
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


# ---------------------------------------------------#
#   Get each box and its score
# ---------------------------------------------------#
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    # Adjust the predicted value to the real value
    # -1,13,13,3,2; -1,13,13,3,2; -1,13,13,3,1; -1,13,13,3,80
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    # Set box_xy, and box_wh to y_min,y_max,xmin,xmax
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    # Score and box
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


# ---------------------------------------------------#
#   Image prediction
# ---------------------------------------------------#
def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    # Get the number of feature layers
    num_layers = len(yolo_outputs)
    # The anchor corresponding to feature layer 1 is 678
    # The anchor corresponding to feature layer 2 is 345
    # The anchor corresponding to feature layer 3 is 012
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    # Process each feature layer for processing
    for e in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[e], anchors[anchor_mask[e]], num_classes, input_shape,
                                                    image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    # Stack the results for each feature layer
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # Retrieve all boxes for box_scores, score_threshold, and scores
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        # remove those with high degree of box overlap
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)


        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_
