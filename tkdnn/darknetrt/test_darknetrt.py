import sys
from ctypes import *

import cv2
import numpy as np
import yaml

lib = CDLL("/root/tkdnn/build/libtrt2.so", RTLD_GLOBAL)

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

free_image = lib.free_image
free_image.argtypes = [IMAGE]

load_network = lib.load_network
load_network.argtypes = [c_char_p, c_int, c_int, c_float]
load_network.restype = c_void_p

class BOX(Structure):
    _fields_ = [("confidence", c_float),
                ("x1", c_float),
                ("y1", c_float),
                ("x2", c_float),
                ("y2", c_float),
                ("cls_id", c_int)]

class DetResult(Structure):
    _fields_ = [("num_boxes", c_int),
                ("Box", POINTER(BOX))]

detect_img = lib.detect_img
detect_img.argtypes = [c_void_p, IMAGE]
detect_img.restype = POINTER(DetResult)


def main():
    rt_weight = "/root/qdsb/qdsb.rt"
    num_classes = 8
    batch_size = 1
    thresh = 0.25
    network = load_network(rt_weight.encode(), num_classes, batch_size, thresh)

    import cv2
    import os
    from pathlib import Path
    from tqdm import tqdm
    root = "/root/qdsb"
    allfiles = list(os.listdir(root))
    for _ in range(1000):
        for file in tqdm(allfiles):
            if Path(file).suffix.lower() not in ['.jpg','.png','.jpeg']:
                continue
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            img_h, img_w = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            darknet_image = make_image(img_w, img_h, 3)
            copy_image_from_bytes(darknet_image, img.tobytes())
            results = detect_img(network, darknet_image)[0]
            free_image(darknet_image)
            num_boxes = results.num_boxes
            for i in range(num_boxes):
                print(results.Box[i].x1, results.Box[i].y1, results.Box[i].x2, results.Box[i].y2)
                x1 = int(results.Box[i].x1)
                y1 = int(results.Box[i].y1)
                x2 = int(results.Box[i].x2)
                y2 = int(results.Box[i].y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255),5)
                print('conf', results.Box[i].confidence, 'class_id: ', results.Box[i].cls_id)
            cv2.imwrite(f'{file}', img)
            import time
            time.sleep(1)
            # break
        break

        

if __name__ == "__main__":
    main()