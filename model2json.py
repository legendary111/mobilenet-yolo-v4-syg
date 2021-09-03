import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from nets.yolo4 import yolo_body
from nets.loss import yolo_loss
from keras.backend.tensorflow_backend import set_session
from utils.utils import get_random_data,get_random_data_with_Mosaic,rand,WarmUpCosineDecayScheduler

Inputs = Input([416, 416, 3])
model = yolo_body(Inputs, 3, 4)
model.summary()

model_json = model.to_json()
with open('model.json', 'w') as file:
    file.write(model_json)
