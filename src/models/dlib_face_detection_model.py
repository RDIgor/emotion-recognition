import dlib


class DlibFaceDetectionModel:
    def __init__(self, weights):
        self.face_detector = dlib.cnn_face_detection_model_v1(weights)

    def predict(self, image):
        dlib_rects = self.face_detector(image, 0)

        rects = []
        for dlib_rect in dlib_rects:
            x = dlib_rect.rect.left()
            y = dlib_rect.rect.top()
            width = dlib_rect.rect.right() - x
            height = dlib_rect.rect.bottom() - y

            rects.append((x, y, width, height))

        return rects

