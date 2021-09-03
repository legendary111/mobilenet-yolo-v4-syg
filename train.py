import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from nets.yolo4 import yolo_body
from nets.loss import yolo_loss
from utils.utils import get_random_data, get_random_data_with_Mosaic, WarmUpCosineDecayScheduler

# Specify GPU and Set the percentage of GPU used
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
session = tf.Session(config=config)
KTF.set_session(session)


# ---------------------------------------------------#
#   Get the class and the prior box
# ---------------------------------------------------#
def get_classes(classes_path):
    """loads the classes"""
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    """loads the anchors from a file"""
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


# ---------------------------------------------------#
#   Training data generator
# ---------------------------------------------------#
def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, mosaic=False):
    """data generator for fit_generator"""
    n = len(annotation_lines)
    i = 0
    flag = True
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            if mosaic:
                if flag and (i + 4) < n:
                    image, box = get_random_data_with_Mosaic(annotation_lines[i:i + 4], input_shape)
                    i = (i + 1) % n
                else:
                    image, box = get_random_data(annotation_lines[i], input_shape)
                    i = (i + 1) % n
                flag = bool(1 - flag)
            else:
                image, box = get_random_data(annotation_lines[i], input_shape)
                i = (i + 1) % n
            image_data.append(image)
            box_data.append(box)
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


# ---------------------------------------------------#
#   Read the XML file and print y_true
# ---------------------------------------------------#
def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    #  three characteristic layers
    num_layers = len(anchors) // 3

    # 678 142,110,  192,243,  459,401
    # 345 36,75,  76,55,  72,146
    # 012 12,16,  19,36,  40,28
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')  # 416,416
    # Read the xy axis, read the length and width
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    # Calculate percentage
    true_boxes[..., 0:2] = boxes_xy / input_shape[:]
    true_boxes[..., 2:4] = boxes_wh / input_shape[:]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[e] for e in range(num_layers)]
    y_true = [np.zeros((m, grid_shapes[e][0], grid_shapes[e][1], len(anchor_mask[e]), 5 + num_classes),
                       dtype='float32') for e in range(num_layers)]
    # [1,9,2]
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # Each graph
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0:
            continue
        # [n,1,2]
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        # Calculate which real box fits the prior box best
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for e in range(num_layers):
                if n in anchor_mask[e]:
                    # Floor is used to round down
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[e][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[e][0]).astype('int32')
                    k = anchor_mask[e].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[e][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[e][b, j, i, k, 4] = 1
                    y_true[e][b, j, i, k, 5 + c] = 1

    return y_true


# ----------------------------------------------------#

# ----------------------------------------------------#
if __name__ == "__main__":
    annotation_path = '2007_train.txt'
    classes_path = 'model_data/car_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    # ------------------------------------------------------#
    #  
    #   Training your own data set prompts dimension mismatches
    #   The predicted things are different ，so the natural dimensions don't match
    # ------------------------------------------------------#
    weights_path = 'model_data/last1.h5'
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names)
    num_anchors = len(anchors)
    log_dir = 'logs/000/'
    # You can use 416x416 for smaller video memory
    # You can use 608x608 for smaller video memory
    input_shape = (416, 416)
    mosaic = True
    Cosine_scheduler = False
    label_smoothing = 0

    K.clear_session()


    image_input = Input(shape=(None, None, 3))
    h, w = input_shape

    # Create the YOLO model
    print('Create YOLOv4 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    model_body = yolo_body(image_input, num_anchors // 3, num_classes)

    # Load the pre-training weights
    print('Load weights {}.'.format(weights_path))
    model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)


    # 26,26,3,85
    # 52,52,3,85
    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[i], w // {0: 32, 1: 16, 2: 8}[i], num_anchors // 3,
                           num_classes + 5)) for i in range(3)]

    # input:*model_body.input, *y_true
    # output:model_loss
    loss_input = [*model_body.output, *y_true]
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5,
                                   'label_smoothing': label_smoothing})(loss_input)

    model = Model([model_body.input, *y_true], model_loss)

    # Training Parameter Setting
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)

    # 0.1 for verification and 0.9 for training
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # ------------------------------------------------------#
    #   The main feature extraction network features are universal and freezing training can speed up training
    #   It can also prevent weights from being destroyed at the beginning of training.
    #   If OOM or video memory is insufficient, adjust Batch_size
    # ------------------------------------------------------#
    freeze_layers = 249
    for i in range(freeze_layers):
        model_body.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))

    # Adjust the non-trunk model
    if True:
        Init_epoch = 0
        Freeze_epoch = 25
        batch_size = 8
        learning_rate_base = 1e-3
            warmup_epoch = int((Freeze_epoch - Init_epoch) * 0.2)
            total_steps = int((Freeze_epoch - Init_epoch) * num_train / batch_size)

            warmup_steps = int(warmup_epoch * num_train / batch_size)
            
            reduce_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                                   total_steps=total_steps,
                                                   warmup_learning_rate=1e-4,
                                                   warmup_steps=warmup_steps,
                                                   hold_base_rate_steps=num_train,
                                                   min_learn_rate=1e-6
                                                   )
            model.compile(optimizer=Adam(), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        else:
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
            model.compile(optimizer=Adam(learning_rate_base), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
            data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes, mosaic=mosaic),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes,
                                           mosaic=False),
            validation_steps=max(1, num_val // batch_size),
            epochs=Freeze_epoch,
            initial_epoch=Init_epoch,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    for i in range(freeze_layers):
        model_body.layers[i].trainable = True

    # Post-thawing training
    if True:
        Freeze_epoch = 25
        Epoch = 50
        batch_size = 2

        learning_rate_base = 1e-4
        if Cosine_scheduler:

            warmup_epoch = int((Epoch - Freeze_epoch) * 0.2)

            total_steps = int((Epoch - Freeze_epoch) * num_train / batch_size)

            warmup_steps = int(warmup_epoch * num_train / batch_size)
 
            reduce_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                                   total_steps=total_steps,
                                                   warmup_learning_rate=1e-5,
                                                   warmup_steps=warmup_steps,
                                                   hold_base_rate_steps=num_train // 2,
                                                   min_learn_rate=1e-6
                                                   )
            model.compile(optimizer=Adam(), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        else:
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
            model.compile(optimizer=Adam(learning_rate_base), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
            data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes, mosaic=mosaic),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes,
                                           mosaic=False),
            validation_steps=max(1, num_val // batch_size),
            epochs=Epoch,
            initial_epoch=Freeze_epoch,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'last1.h5')
