
import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

# prediction
if __name__ == "__main__":
    yolo = YOLO()
    
    # 'predict' : Single picture prediction
    # 'fps' : calculate fps
    mode = "predict"

    crop            = False
    count           = False


    test_interval   = 100
    fps_image_path  = "img/xx.jpg"

    # Single picture prediction
    if mode == "predict":

        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop = crop, count=count)
                r_image.show()

    # calculate fps
    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
                 
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'fps'.")
