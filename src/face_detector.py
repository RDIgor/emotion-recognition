from models.retina_face_model import RetinaFaceModel
from utils import clip_rects


class FaceDetector:
    def __init__(self):
        # load face detection model
        self.face_model = RetinaFaceModel("/opt/R50-model/", 0, -1, 'net3')
        pass

    def detect(self, frame):
        # detect faces
        (faces, landmarks) = self.face_model.predict(frame)

        # clip faces
        clipped_faces = clip_rects(frame, faces)

        return clipped_faces, landmarks
