import dlib
import cv2
import imutils
import numpy as np
from src.models.base_model import BaseModel


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


class FaceLandmarkPredictionModel(BaseModel):
    def __init__(self, model_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)

    # return list of points, [(x, y)]
    def predict(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        x_scale = 1#image.shape[1] / image.shape[1]
        y_scale = 1#image.shape[0] / image.shape[0]

        (h, w) = gray.shape

        dlib_rect = dlib.rectangle(
                int(0),
                int(0),
                int(w / x_scale),
                int(h / y_scale))

        shape = self.predictor(gray, dlib_rect)

        shape = shape_to_numpy(shape, x_scale, y_scale)

        return shape
