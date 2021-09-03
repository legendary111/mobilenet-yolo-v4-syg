from yolo import YOLO
from PIL import Image

yolo = YOLO()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')

        continue
    else:
        r_image = yolo.detect_image(image)
        # r_image.show()
        r_image.save(img[0:2] + '_re.jpg')
yolo.close_session()
