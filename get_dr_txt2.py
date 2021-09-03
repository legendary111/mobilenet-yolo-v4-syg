# ----------------------------------------------------#
#   获取测试集的detection-result和images-optional
#   具体视频教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
# ----------------------------------------------------#
from yolo import YOLO
from PIL import Image
from keras.layers import Input
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
from utils.utils import letterbox_image
from keras.models import load_model
from nets.yolo4 import yolo_body, yolo_eval
import colorsys
import numpy as np
import os


class mAP_YOLO(YOLO):
    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self):
        # self.score = 0.05
        self.score = 0.3
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # 计算anchor数量
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        # 载入模型，如果原来的模型里已经包括了模型结构则直接载入。
        # 否则先构建模型再载入
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                   'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        # 打乱颜色
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        self.input_image_shape = K.placeholder(shape=(2,))

        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           num_classes, self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image_id, image):
        f = open(save_path + "input/detection-results/" + image_id + ".txt", "w")
        # 调整图片使其符合输入要求
        boxed_image = letterbox_image(image, self.model_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # 预测结果
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        for i, c in enumerate(out_classes):
            predicted_class = self.class_names[int(c)]
            score = str(out_scores[i])

            top, left, bottom, right = out_boxes[i]
            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return


yolo = mAP_YOLO()

all_path = 'D:\\0VOC\\VOCdevkit0920图像和xml备份-3dataset-9278\\'
save_path = 'D:\\PythonCode\\5test-v4-mobile\\'
image_ids = open(all_path + 'test.txt').read().strip().split()

if not os.path.exists(save_path + "input"):
    os.makedirs(save_path + "input")
if not os.path.exists(save_path + "input/detection-results"):
    os.makedirs(save_path + "input/detection-results")
if not os.path.exists(save_path + "input/images-optional"):
    os.makedirs(save_path + "input/images-optional")

num_img = 0
for image_id in image_ids:
    num_img += 1
    print('num_img=', num_img)
    image_path = all_path + "JPEGImages/" + image_id + ".jpg"
    image = Image.open(image_path)
    # 开启后在之后计算mAP可以可视化
    image.save(save_path + "input/images-optional/" + image_id + ".jpg")
    yolo.detect_image(image_id, image)
    print(image_id, " done!")

print("Conversion completed!")
