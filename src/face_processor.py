from face_aligner import FaceAligner
from models.face_landmark_prediction_model import FaceLandmarkPredictionModel
from utils import crop_image


class FaceProcessor:
    def __init__(self, config):
        # load landmarks model
        self.landmark_model = FaceLandmarkPredictionModel(config["face_landmark_detection"]["weights"])

    def process(self, frame, face_box, face_five_landmarks):
        # crop face
        face_image = crop_image(frame, face_box)

        # align face image
        # aligned_face_image = FaceAligner.align(face_image, face_five_landmarks)

        # detect landmarks
        return self.landmark_model.predict(face_image)
