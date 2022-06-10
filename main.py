import cv2
import os
from sklearn.cluster import KMeans
import numpy as np
from time import time
# from pyproj import Proj
# from pyproj import transform
import shutil
from paddleocr import PaddleOCR  # USE GPU
from scipy import ndimage
from keras.applications.vgg16 import VGG16
from sklearn.cluster import KMeans
from random import randint
import string
import pickle
from detection_class import MserDetection
from local_class import Loclization
from letter_shape_class import *

list_x = []


def main():
    parent_file = r"pictures_to_process"
    count = 0
    while True:
        try:
            list_of_pictures = [x for x in os.listdir(parent_file) if ".txt" not in x][1:40]
        except FileNotFoundError:
            continue
        for i in list_of_pictures:
            loaded_image = cv2.imread(os.path.join(parent_file, i))
            mser_obj = MserDetection(loaded_image, 0, 1000, 500, 100000, 5, None)
            box = mser_obj.mser_create(False)
            for j in box:
                try:
                    x_min, y_min, x_max, y_max = j
                    img = loaded_image[y_min - 50:y_max + 50, x_min - 50:x_max + 50]
                    cv2.imshow("img", img)
                    cv2.waitKey(0)
                except Exception:
                    print("Error")
                    continue
                try:
                    classification = Classification(img)
                    dicti = classification.classify()
                except BrokenPipeError:
                    print("Error detection error")
                    print("Continuing")
                    continue
                except ValueError:
                    print("Value detection error")
                    print("Continuing")
                    continue

                prompt = int(input("enter a 0 to cancel 1 to continue"))
                localization_obj = Loclization(center_of_y=(y_min + int(abs(y_min - y_max) / 2)),
                                               center_of_x=(x_min + int(abs(x_min - x_max) / 2)),
                                               gps_drone="file_path_of_telemntary data",
                                               oriantaion=classification.get_oriantation())
                long, lat = localization_obj.object_location()
                orintation = localization_obj.oriantation()
                if not prompt:
                    continue
                    # IMPORTANT !! DON'T FORGET TO COMPLETE THE CODE
                with open('readme.txt', 'w') as file:
                    file.write(dicti["object_letter"])
                    file.write(orintation)
                    file.write("Thing okay")


if __name__ == "__main__":
    main()
