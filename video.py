# -------------------------------------#
#       Camera Detection
# -------------------------------------#
from yolo import YOLO
from PIL import Image
import numpy as np
import cv2
import time
yolo = YOLO()


# capture=cv2.VideoCapture(0) #
capture = cv2.VideoCapture("a1.mp4")

i = 0 
fps = 0.0
while(True):
    t1 = time.time()
    # Read a frame
    ref, frame = capture.read()
    # Format transformation, BGRtoRGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = Image.fromarray(np.uint8(frame))

    # testing
    frame = np.array(yolo.detect_image(frame))

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    fps = (fps + (1./(time.time()-t1))) / 2
    print("fps= %.2f" % fps)
    frame = cv2.putText(frame, "fps= %.2f" % fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    i = i+1
    cv2.imwrite(str(i)+'.jpg', frame)
    # cv2.imshow("video",frame)
    c = cv2.waitKey(1) & 0xff
    if c == 27:
        capture.release()
        break

yolo.close_session()    
