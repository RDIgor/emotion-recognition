import numpy as np
from src.models.emotion_recognition_model import EmotionRecognitionModel
from src.common import Emotion


class EmotionProcessor:
    def __init__(self, config):
        self.smile_model = EmotionRecognitionModel(config['emotion_recognition']['weights'])

    def process(self, landmarks, frame):
        # extract mouth landmarks
        mouth_landmarks = self.extract_mouth_landmarks(landmarks)

        # image shape
        h, w = frame.shape[:2]

        normalized_mouth_landmarks = []
        for (x, y) in mouth_landmarks:
            normalized_mouth_landmarks.append(float(x) / w)
            normalized_mouth_landmarks.append(float(y) / h)

        prediction = self.smile_model.predict([normalized_mouth_landmarks])

        emotion = Emotion(prediction)

        return emotion

    @staticmethod
    def extract_mouth_landmarks(landmarks):
        mouth_landmarks = landmarks[180:194]
        mouth_landmarks = np.concatenate((mouth_landmarks, landmarks[11:21]))
        mouth_landmarks = np.concatenate((mouth_landmarks, landmarks[22:26]))
        mouth_landmarks = np.concatenate((mouth_landmarks, landmarks[152:180]))

        return mouth_landmarks
