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

shape_encoding = {0: 1, 1: 13, 2: 10, 3: 9, 4: 11, 5: 8, 6: 3, 7: 6, 8: 1,
                  9: 5, 10: 12, 11: 7, 12: 4}

shape_encoding_1 = {1: "Circle", 2: "SEMI CIRCLE", 3: "QUARTER CIRCLE",
                    4: "Triangle", 5: "SQAURE", 6: "RECTANGLE", 7: "TRAPEZOID",
                    8: "PENTAGON", 9: "HEXAGON", 10: "HEPTAGON", 11: "OCTAGON",
                    12: "STAR", 13: "CROSS"}

color_encoding = {"WHITE": 1, "BLACK": 2, "GRAY": 3, "RED": 4, "BLUE": 5,
                  "GREEN": 6, "YELLOW": 7, "PURPLE": 8, "BRONW": 9,
                  "ORANGE": 10}

shape_model = pickle.load(open(r"shape_model_pickle", 'rb'))
color_model = pickle.load(open(r"knn_file", 'rb'))
ocr = PaddleOCR(lang='en')

model_vgg = VGG16(include_top=False,
                  weights="imagenet",
                  input_shape=(150, 150, 3))

for layers in model_vgg.layers:
    layers.trainable = False


def k_means_algorithm(cropped_blurred):
    time_start = time()
    k_m_image = cropped_blurred.reshape((-1, 3))
    k_m_image = np.float32(k_m_image)
    param = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 10, 1.0)
    k = 2
    try:
        ret, label, center = cv2.kmeans(k_m_image, k, None, param, 10, cv2.KMEANS_RANDOM_CENTERS)
    except:
        return
    center = np.uint8(center)
    np.unique(label)
    print(center)
    print("The label is = ", np.unique(label))
    print("The colors is = ", center)
    res = center[label.flatten()]
    k_means = res.reshape(cropped_blurred.shape)
    print("Time it takes for k_means: ", time() - time_start)
    k_means = cv2.resize(k_means, (240, 144))
    cv2.imshow('k_means', k_means)
    cv2.waitKey(0)
    return k_means, center


def k_means(blur_image, n_clusters=3):
    x, y, z = blur_image.shape
    image_2d = blur_image.reshape(x * y, z)
    k_means_cluster = KMeans(n_clusters=n_clusters)
    k_means_cluster.fit(image_2d)
    center = k_means_cluster.cluster_centers_
    label = k_means_cluster.labels_
    return center[label].reshape(x, y, z), center, label


class Classification:
    image: np.array
    blured_image: np.array
    gray_image: np.array
    k_image = np.array
    object_shape: str
    object_letter: str
    object_letter_color: str
    object_shape_color: str
    object_orintation: int
    __letter = []
    predicted_colors = []

    __dict__ = {"object_shape": None,
                "object_letter": None,
                "object_letter_color": None,
                "object_shape_color": None,
                "object_oriantation": None}

    def __init__(self, image):
        self.image = image
        self.blured_image = cv2.GaussianBlur(self.image, ksize=(11, 11), sigmaX=0)
        self.k_image = k_means(self.blured_image)
        self.foreground = self.gather_shape_mask()
        self.masked_letter = self.gather_letter_mask()

    def get_shape(self):
        return self.__dict__["object_shape"]

    def get_letter(self):
        return self.__dict__["object_letter"]

    def get_oriantation(self):
        return self.__dict__["object_oriantation"]

    def get_letter_color(self):
        return self.__dict__["object_letter_color"]

    def get_shape_color(self):
        return self.__dict__["object_shape_color"]

    def gather_shape_mask(self):  # returns the shape masked using floor flid :)
        cropped_image_blur = cv2.cvtColor(cv2.GaussianBlur(self.blured_image, (5, 5), 0), cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(cropped_image_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh_copy = thresh.copy()
        mask = np.zeros((thresh.shape[0] + 2, thresh.shape[1] + 2), np.uint8)
        cv2.floodFill(thresh_copy, mask, (0, 0), 255)
        im_out = cv2.morphologyEx(thresh | cv2.bitwise_not(thresh_copy), cv2.MORPH_OPEN,
                                  np.ones(shape=(5, 5), dtype=np.uint8))
        if np.mean(im_out) > 230 or np.mean(im_out) < 50:
            print("The image is just white or black exiting")
            raise BrokenPipeError
        im_out = cv2.erode(im_out, np.ones((3, 3), dtype=np.uint8), iterations=3)
        im_out = cv2.dilate(im_out, np.ones((3, 3), dtype=np.uint8), iterations=2)
        im_out = cv2.erode(im_out, np.ones((3, 3), dtype=np.uint8), iterations=3)
        cv2.imshow('Foreground', im_out)
        cv2.waitKey(0)
        return im_out

    def shape_classify(self):
        img = cv2.resize(self.foreground, (150, 150))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imshow("img", img)
        cv2.waitKey(0)
        print(img.shape, "Shape of image")
        input_img = np.expand_dims(img, axis=0)  # Expand dims so the input is (num images, x, y, c)
        print(input_img.shape)
        feature_extractor = model_vgg.predict(input_img)
        features = feature_extractor.reshape(feature_extractor.shape[0], -1)
        y = shape_model.predict(features)
        shape_name = shape_encoding_1[shape_encoding[y[0]]]
        print("The shape is:", shape_name)
        print(y)
        return shape_encoding[y]

    def gather_letter_mask(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        background = cv2.erode(self.foreground, np.ones(shape=(7, 7), dtype=np.uint8))
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        mask = cv2.bitwise_and(thresh, thresh, mask=background)  # FOR COLOR DETECTION
        cv2.imshow('mask', mask)
        cv2.waitKey(0)
        return mask

    def __letter_verify(self, angle_of_rotation):  # put letter verifyication or teplate
        rotated = ndimage.rotate(self.masked_letter, angle_of_rotation)
        result = ocr.ocr(rotated, det=False, cls=False)
        cv2.imshow("picture", rotated)
        cv2.waitKey(0)
        print(result)
        return (result, True) if result[0][1] > 0.40 and result[0][0][0].isalnum() else (("", 0), False)

    def letter_classify(self):
        max_accuracy, char, angle = 0, "", 0
        avg, cnt_of_fn = 0, 0  # Average, count of false negatives
        for i in range(0, 360, 15):
            result, boolean = self.__letter_verify(i)
            print(result, "The result of letter classification")
            print("The result is:", result)
            print("The boolean is:", boolean)
            try:
                current_char, current_accuracy = result
            except ValueError:
                current_char, current_accuracy = result[0]
            avg += current_accuracy
            max_accuracy = current_accuracy if max_accuracy < current_accuracy else max_accuracy
            char = current_char if max_accuracy == current_accuracy else char
            angle = i if max_accuracy == current_accuracy else angle
            print("The chosen char is:", char)
            print("The accuracy is:", max_accuracy)
            cnt_of_fn += 0 if boolean else 1
            print(cnt_of_fn, "The false negative")
            if cnt_of_fn > (360 / 18):
                raise BrokenPipeError
        print(max_accuracy, char, angle)
        self.object_orintation = angle
        return None if avg / (360 / 15) < 50 else max_accuracy, char[0], angle

    def get_shape_color(self):  # if you get error here then check the encoding
        img, center, label = self.k_image
        percent = []
        colors = []
        label = label.tolist()
        for i in range(len(center)):
            j = label.count(i)
            j = j / (len(label))
            percent.append(j)
            colors.append(color_model.predict(center[i].reshape(1, -1)))
        colors = [color_encoding[x] for x in colors]
        return colors[0] if colors[0] not in self.predicted_colors else colors[1] if colors[
                                                                                         1] not in self.predicted_colors \
            else colors[2]

    def get_letter_color(self):
        print(self.k_image[0].shape)
        print(self.masked_letter.shape)
        masked_letter_image = cv2.resize(self.masked_letter, (200, 200))
        k_means_image = cv2.resize(self.k_image[0], (200, 200))
        masked_letter_k_means = cv2.bitwise_and(k_means_image, k_means_image, mask=masked_letter_image)
        cv2.imshow("color_picture", masked_letter_k_means)
        cv2.waitKey(0)
        img, center, label = k_means(masked_letter_k_means, 2)
        print(len(np.unique(label)), "How many labels ")
        label = label.tolist()
        print("The unique elements in the img")
        print(np.unique(img))
        print()
        percent = []
        self.predicted_colors = []
        for i in range(len(center)):
            j = label.count(i)
            j = j / (len(label))
            percent.append(j)
            self.predicted_colors.append(color_model.predict(center[i].reshape(1, -1)))
        if percent[0] > percent[1]:
            print(self.predicted_colors[1])
            return color_encoding[self.predicted_colors[1]]
        print(self.predicted_colors[0])
        return color_encoding[self.predicted_colors[0]]

    def classify(self):
        self.__dict__["object_letter"] = self.letter_classify()[1]
        self.__dict__["object_oriantation"] = self.object_orintation
        self.__dict__["object_shape"] = self.shape_classify()
        self.__dict__["object_letter_color"] = self.get_letter_color()
        self.__dict__["object_shape_color"] = self.get_shape_color()
        print(self.__dict__)
        return self.__dict__
