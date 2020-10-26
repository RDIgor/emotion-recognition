import cv2


class FaceDetectionModel:
    def __init__(self, config_path, model_path, confidence_threshold=0.4, device="cpu"):
        self.net = cv2.dnn.readNet(model_path, config_path)

        if device == "cpu":
            self.net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        elif device == "gpu":
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.confidence_threshold = confidence_threshold

    def predict(self, image):
        (height, width) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)

        self.net.setInput(blob)

        detections = self.net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)
                boxes.append((x1, y1, x2 - x1, y2 - y1))

        return boxes
