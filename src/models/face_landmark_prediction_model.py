import dlib
import cv2
import imutils
import numpy as np


def shape_to_numpy(shape, x_scale, y_scale, dtype="int"):
    coordinates = np.zeros((shape.num_parts, 2), dtype=dtype)

    for i in range(0, shape.num_parts):
        coordinates[i] = (shape.part(i).x * x_scale, shape.part(i).y * y_scale)

    return coordinates


def dlib_rect_to_box(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return x, y, w, h


class FaceLandmarkPredictionModel:
    def __init__(self, model_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)

    def predict(self, image, rects):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        x_scale = 1#image.shape[1] / image.shape[1]
        y_scale = 1#image.shape[0] / image.shape[0]

        dlib_rects = []

        for rect in rects:
            dlib_rects.append(dlib.rectangle(
                int(rect[0] / x_scale),
                int(rect[1] / y_scale),
                int((rect[0] + rect[2]) / x_scale),
                int((rect[1] + rect[3]) / y_scale)))

        result = []
        for (i, dlib_rect) in enumerate(dlib_rects):
            shape = self.predictor(gray, dlib_rect)

            shape = shape_to_numpy(shape, x_scale, y_scale)

            result.append(shape)

        return result
