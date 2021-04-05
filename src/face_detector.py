from src.models.retina_face_model import RetinaFaceModel
from src.utils import clip_rects


class FaceDetector:
    def __init__(self, config):
        # load model according to the config

        # load face detection model
        self.face_model = RetinaFaceModel("/opt/R50-model/", 0, 0, 'net3')

    def detect(self, frame):
        # detect faces
        (faces, landmarks) = self.face_model.predict(frame)

        return clip_rects(frame, faces), landmarks
