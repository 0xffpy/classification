import cv2
import os
import numpy as np
from time import time
import string
from random import randint
import shutil


def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("int")


class MserDetection:
    __MSER_OBJECT = cv2.MSER_create()
    _min_length: int = 0
    _max_length: int = 0
    _ratio = 0
    _file_of_images = ""  # input of class to this spesfic file

    def __init__(self, image, _min_length, _max_length, _min_area, _max_area, ratio, file_of_images):
        self._file_of_images = file_of_images
        self._min_length = _min_length
        self._max_length = _max_length
        self._ratio = ratio
        self.image = image.copy()
        self.image_resized = cv2.resize(image, (1200, 800))
        self.gray_image = cv2.cvtColor(self.image_resized, cv2.COLOR_BGR2GRAY)
        self.blur_image = cv2.GaussianBlur(self.gray_image, ksize=(15, 15), sigmaX=0)
        self.__MSER_OBJECT.setMaxArea(_max_area)
        self.__MSER_OBJECT.setMinArea(_min_area)

    def mser_create(self, is_bounding=True):
        time_start = time()
        regions, _ = self.__MSER_OBJECT.detectRegions(self.blur_image)
        boxes = []
        print(len(regions))
        for p in regions:
            x_max, y_max = np.amax(p, axis=0)
            x_min, y_min = np.amin(p, axis=0)
            try:
                if not (abs(x_max - x_min) > self._max_length or abs(y_min - y_max) > self._max_length):
                    if not (abs(x_max - x_min) < self._min_length or abs(y_min - y_max) < self._min_length):
                        if not (abs(y_min - y_max) / abs(x_max - x_min) > self._ratio or
                                abs(x_max - x_min) / abs(y_min - y_max) > self._ratio):
                            boxes.append((x_min, y_min, x_max, y_max))
            except RuntimeError:
                continue

        box = non_max_suppression_fast(np.array(boxes), 0.1)
        total_time = time() - time_start
        if is_bounding:
            for i in box:
                x_min, y_min, x_max, y_max = i
                cv2.rectangle(self.image_resized, (x_min, y_min), (x_max, y_max), color=(0, 0, 0),
                              thickness=5)
            cv2.imshow("Detection", self.image_resized)
            cv2.waitKey(0)
        print("The time it took: ", total_time)
        return box * 5