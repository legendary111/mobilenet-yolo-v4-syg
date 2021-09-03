from yolo import YOLO, detect_video
from PIL import Image
import os


# Specify GPU and Set the percentage of GPU used
# import os
# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
# session = tf.Session(config=config)
# KTF.set_session(session)
# def detect_img(img, yolo):
    r_image = yolo.detect_image(img)
    return r_image


if __name__ == '__main__':
    # class YOLO defines the default value
    yolo = YOLO()
    FLAG = False
    if FLAG is True:
        for (root, dirs, files) in os.walk('img'):
            if files:
                for f in files:
                    print(f)
                    path = os.path.join(root, f)
                    image = Image.open(path)
                    image = detect_img(image, yolo)
                    image.save('img/res/'+f)
        yolo.close_session()
    else:
        video_path = "video/0820_new.mp4"
        video_save_path = "video/0829dataset-4class/res_0820_new.mp4"
        detect_video(yolo, video_path, video_save_path)
