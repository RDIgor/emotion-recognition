import numpy as np


class EmotionRecognitionModel:
    def __init__(self, path):
        import keras
        self.model = keras.models.load_model(path)

    def predict(self, landmarks):
        output = self.model.predict(np.asarray(landmarks).astype('float64'))

        return np.argmax(output[0])
