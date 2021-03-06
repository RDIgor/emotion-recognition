import cv2
import numpy as np
from src.models.base_model import BaseModel


class CvYoloModel(BaseModel):
    def __init__(self, cfg_path, weights_path, confidence_threshold=0.3, nms_threshold=0.4):
        self.net = cv2.dnn.readNet(weights_path, cfg_path)
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.net_width = 416
        self.net_height = 416
        layers = self.net.getLayerNames()
        self.ln = [layers[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def predict(self, image):
        (height, width) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (self.net_width, self.net_height), swapRB=True, crop=False)

        self.net.setInput(blob)

        outputs = self.net.forward(self.ln)

        return self.extract_boxes(outputs, width, height)

    def extract_boxes(self, outputs, width, height):
        boxes = []
        confidences = []
        class_ids = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.confidence_threshold:
                    box = detection[0:4] * np.array([width, height, width, height])

                    (center_x, center_y, w, h) = box.astype("int")
                    x = int(center_x - (w / 2))
                    y = int(center_y - (h / 2))

                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        ids = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

        result = []  # x, y, w, h
        if len(ids) > 0:
            for i in ids.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                result.append((x, y, w, h, class_ids[i], confidences[i]))

        return result
