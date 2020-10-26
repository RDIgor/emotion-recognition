import dlib
import cv2
import imutils
import numpy as np


def shape_to_numpy(shape, x_scale, y_scale, dtype="int"):
    coordinates = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x * x_scale, shape.part(i).y * y_scale)

    return coordinates


def dlib_rect_to_box(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return x, y, w, h


class FaceLandmarkDetectionModel:
    def __init__(self, model_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)

    def predict(self, image):
        resized = imutils.resize(image, width=500)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        x_scale = image.shape[1] / resized.shape[1]
        y_scale = image.shape[0] / resized.shape[0]

        rects = self.detector(gray, 1)

        result = []
        for (i, rect) in enumerate(rects):
            shape = self.predictor(gray, rect)

            shape = shape_to_numpy(shape, x_scale, y_scale)

            (x, y, w, h) = (dlib_rect_to_box(rect) * np.array([x_scale, y_scale, x_scale, y_scale])).astype("int")

            dictionary = {(x, y, w, h): shape}

            result.append(dictionary)

        return result
