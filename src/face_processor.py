from face_aligner import FaceAligner
from models.face_landmark_prediction_model import FaceLandmarkPredictionModel


class FaceProcessor:
    def __init__(self, config):
        # load landmark model
        self.landmark_model = FaceLandmarkPredictionModel(config["face_landmark_detection"]["weights"])

    def process(self, frame, face, face_five_landmarks):
        # crop face
        (face_x, face_y, face_w, face_h) = face
        face_image = frame[int(face_y): int(face_y + face_h), int(face_x): int(face_x + face_w)]

        # remap five landmarks
        face_remapped_five_landmarks = []
        for landmark in face_five_landmarks:
            face_remapped_five_landmarks.append((landmark[0] - face_x, landmark[1] - face_y))

        # align face image
        aligned_face_image = FaceAligner.align(face_image, face_remapped_five_landmarks)

        # detect landmarks
        face_landmarks = self.landmark_model.predict(face_image)

        # remap landmarks for original image
        result_face_landmarks = []

        for (x, y) in face_landmarks:
            result_face_landmarks.append((face_x + x, face_y + y))

        return result_face_landmarks
