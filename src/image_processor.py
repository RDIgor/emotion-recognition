from src.face_detector import FaceDetector
from src.face_processor import FaceProcessor
from src.http_request_maker import HTTPRequestMaker
from src.common import Face
from src.emotion_processor import EmotionProcessor
from src.utils import crop_image
import cv2


class ImageProcessor:
    def __init__(self, config, args):
        if args.use_server:
            # init ip and port
            self.use_http_server = True
            self.http_request_maker = HTTPRequestMaker(args.ip, args.port)
        else:
            self.use_http_server = False

            # init face detector
            self.face_detector = FaceDetector(config)

            # init face processor
            self.face_processor = FaceProcessor(config)

            # init smile processor
            self.emotion_processor = EmotionProcessor(config)

    def process(self, frame):
        (face_boxes, five_landmarks) = self.get_faces(frame)

        faces = []
        for i in range(0, len(face_boxes)):
            face_box = face_boxes[i]
            face_five_landmarks = five_landmarks[i]
            face_landmarks = self.get_landmarks(frame, face_box, face_five_landmarks)
            face_applicability = True
            emotion = None

            # check if point of nose is located in the area on the nose. Nose point index is 2.
            nose_point = (face_five_landmarks[2][0] - face_box[0], face_five_landmarks[2][1] - face_box[1])
            nose_landmarks = face_landmarks[135:152]

            if cv2.pointPolygonTest(nose_landmarks, nose_point, False) < 0:
                face_applicability = False

            if face_applicability:
                emotion = self.get_emotion(face_landmarks, crop_image(frame, face_box))

            # add face
            faces.append(Face(
                box=face_box,
                landmarks=face_landmarks,
                five_landmarks=face_five_landmarks,
                applicability=face_applicability,
                emotion=emotion))

        return faces

    def get_faces(self, frame):
        if self.use_http_server:
            return self.http_request_maker.faces_GET(frame)
        else:
            return self.face_detector.detect(frame)

    def get_landmarks(self, face_image, face_box, face_five_landmarks):
        if self.use_http_server:
            return self.http_request_maker.landmarks_GET(face_image, face_box, face_five_landmarks)
        else:
            return self.face_processor.process(face_image, face_box, face_five_landmarks)

    def get_emotion(self, landmarks, face_image):
        if self.use_http_server:
            return None
        else:
            return self.emotion_processor.process(landmarks, face_image)
