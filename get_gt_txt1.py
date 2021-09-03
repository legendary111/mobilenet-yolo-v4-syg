# ----------------------------------------------------#
#   获取测试集的ground-truth
#   具体视频教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
# ----------------------------------------------------#
import sys
import os
import glob
import xml.etree.ElementTree as ET

all_path = 'D:\\0VOC\\VOCdevkit0920图像和xml备份-3dataset-9278\\'
save_path = 'D:\\PythonCode\\5test-v4-mobile\\'
image_ids = open(all_path + 'test.txt').read().strip().split()

if not os.path.exists(save_path + "input"):
    os.makedirs(save_path + "input")
if not os.path.exists(save_path + "input/ground-truth"):
    os.makedirs(save_path + "input/ground-truth")

file_num = 0
for image_id in image_ids:
    file_num += 1
    with open(save_path + "input/ground-truth/"+image_id+".txt", "w") as new_f:
        root = ET.parse(all_path + "Annotations/"+image_id+".xml").getroot()
        line = 0
        for obj in root.findall('object'):
            if obj.find('difficult') is not None:
                difficult = obj.find('difficult').text
                if int(difficult) == 1:
                    continue
            obj_name = obj.find('name').text
            bndbox = obj.find('bndbox')
            left = bndbox.find('xmin').text
            top = bndbox.find('ymin').text
            right = bndbox.find('xmax').text
            bottom = bndbox.find('ymax').text
            # new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
            # 删除结尾最后一个空行
            line += 1
            new_f.write("%s %s %s %s %s" % (obj_name, left, top, right, bottom))
            if line < len(root.findall('object')):
                new_f.write('\n')

print("Conversion completed!")
