import tensorflow as tf
from keras import backend as K
from nets.ious import box_ciou
from nets.yolo4 import yolo_head


# ---------------------------------------------------#
#   Smooth label
# ---------------------------------------------------#
def _smooth_labels(y_true, label_smoothing):
    num_classes = tf.cast(K.shape(y_true)[-1], dtype=K.floatx())
    label_smoothing = K.constant(label_smoothing, dtype=K.floatx())
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes


# ---------------------------------------------------#
#   Used to calculate the IOU for each prediction box versus the real box
# ---------------------------------------------------#
def box_iou(b1, b2):
    # 13,13,3,1,4
    # Compute the coordinates of the upper left and lower right corners
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # 1,n,4
    # Compute the coordinates of the upper left and lower right corners
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # Calculated overlap area
    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)
    return iou


# ---------------------------------------------------#
#   Loss value calculation
# ---------------------------------------------------#
def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, label_smoothing=0.1, print_loss=False):
    # three floors
    num_layers = len(anchors) // 3

    y_true = args[num_layers:]
    yolo_outputs = args[:num_layers]

    # priori box
    # 678   142,110,  192,243,  459,401
    # 345   36,75,  76,55,  72,146
    # 012   12,16,  19,36,  40,28
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    #input_shpae  608,608
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))

    loss = 0

    # Take out each picture
    m = K.shape(yolo_outputs[0])[0]
    mf = K.cast(m, K.dtype(yolo_outputs[0]))


    for e in range(num_layers):
        # Extract the position of the point with target in the feature layer
        object_mask = y_true[e][..., 4:5]
        # Take out the corresponding species
        true_class_probs = y_true[e][..., 5:]
        if label_smoothing:
            true_class_probs = _smooth_labels(true_class_probs, label_smoothing)

        # Process the characteristic layer output of YOLO_outputs
        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[e],
                                                     anchors[anchor_mask[e]], num_classes, input_shape, calc_loss=True)

        # Decoded predicted box position
        # (m,13,13,3,4)
        pred_box = K.concatenate([pred_xy, pred_wh])

        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        # Calculate ignore_mask for each image
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[e][b, ..., 0:4], object_mask_bool[b, ..., 0])
            # Calculate the IOU of the prediction against the real situation 
            # 13,13,3,n
            iou = box_iou(pred_box[b], true_box)

            # 13,13,3
            best_iou = K.max(iou, axis=-1)

            # If the degree of overlap between some prediction boxes and real boxes is greater than 0.5, it is ignored.。
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b + 1, ignore_mask

        # Walk through all the images
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])

        # Compress the content of each picture for processing
        ignore_mask = ignore_mask.stack()
        # (m,13,13,3,1)
        ignore_mask = K.expand_dims(ignore_mask, -1)

        box_loss_scale = 2 - y_true[e][..., 2:3] * y_true[e][..., 3:4]

        # Calculate ciou loss as location loss
        raw_true_box = y_true[e][..., 0:4]
        ciou = box_ciou(pred_box, raw_true_box)
        ciou_loss = object_mask * box_loss_scale * (1 - ciou)
        ciou_loss = K.sum(ciou_loss) / mf
        location_loss = ciou_loss

        # Calculate the cross entropy of 1 and confidence if there is a box at the location
        # If the location does not have a box and satisfies best_iou&lt; Ignore_thresh is considered a negative sample
        # best_iou<ignore_thresh     Used to limit the number of negative samples
        confidence_loss = object_mask * \
            K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
            (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) * ignore_mask

        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += location_loss + confidence_loss + class_loss
        # if print_loss:
        # loss = tf.Print(loss, [loss, location_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss:')
    return loss
