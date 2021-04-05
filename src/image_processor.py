from face_detector import FaceDetector
from face_processor import FaceProcessor
from common import Face
import cv2


class ImageProcessor:
    def __init__(self, config):
        # init face detector
        self.face_detector = FaceDetector(config)

        # init face processor
        self.face_processor = FaceProcessor(config)

    def process(self, frame):
        (face_boxes, five_landmarks) = self.face_detector.detect(frame)

        faces = []
        for i in range(0, len(face_boxes)):
            face_box = face_boxes[i]
            face_five_landmarks = five_landmarks[i]
            face_landmarks = self.face_processor.process(frame, face_box, face_five_landmarks)
            face_applicability = True

            # check if point of nose is located in the area on the nose. Nose point index is 2.
            nose_point = tuple(face_five_landmarks[2] - (face_box[0], face_box[1]))
            nose_landmarks = face_landmarks[135:152]

            if cv2.pointPolygonTest(nose_landmarks, nose_point, False) < 0:
                face_applicability = False

            # add face
            faces.append(Face(
                box=face_box,
                landmarks=face_landmarks,
                five_landmarks=face_five_landmarks,
                applicability=face_applicability))

        return faces
