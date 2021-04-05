import cv2
import base64
import requests
import numpy as np


class HTTPRequestMaker:
    def __init__(self, ip: str, port: str):
        # init URLs
        self.faces_url = "http://{}:{}/faces".format(ip, port)
        self.landmarks_url = "http://{}:{}/landmarks".format(ip, port)

    def faces_GET(self, frame):
        encoded, buffer = cv2.imencode(".jpg", frame)

        data = {"frame": base64.b64encode(buffer)}

        result = requests.get(url=self.faces_url,
                              headers={'Content-Type': "application/json", 'Accept': "application/json"},
                              json=data)

        result_json = result.json()

        faces = result_json['faces']
        five_landmarks = result_json['five_landmarks']

        return faces, five_landmarks

    def landmarks_GET(self, face_image, face_box, face_five_landmarks):
        encoded, buffer = cv2.imencode(".jpg", face_image)
        data = {"frame": base64.b64encode(buffer), "face_box": face_box, "face_five_landmarks": face_five_landmarks}

        result = requests.get(url=self.landmarks_url,
                              headers={'Content-Type': "application/json", 'Accept': "application/json"},
                              json=data)

        result_json = result.json()

        return np.array(result_json['landmarks'])

