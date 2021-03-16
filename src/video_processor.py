import cv2
import json
import utils
import numpy as np
from models.face_landmark_prediction_model import FaceLandmarkPredictionModel
from models.retina_face_model import RetinaFaceModel


class VideoProcessor:
    def __init__(self, args):
        # open config
        with open(args["config"]) as json_file:
            self.json_config = json.load(json_file)

        # load face detection model
        self.face_model = RetinaFaceModel("/opt/R50-model/", 0, -1, 'net3')

        # load landmark model
        self.landmark_model = FaceLandmarkPredictionModel(self.json_config["face_landmark_detection"]["weights"])

        # open video file
        self.capture = cv2.VideoCapture(args['input'])

        # open video writer
        # self.video_writer = VideoWriter("output.avi", 640, 640)

        if not self.capture.isOpened():
            print('Video file does not open')

    def start(self):
        while True:
            # read frame
            ret, frame = self.capture.read()

            if not ret:
                print('Failed to read frame')

                return

            # detect faces
            (faces, five_landmarks) = self.face_model.predict(frame)

            # clip faces
            clipped_faces = utils.clip_rects(frame, faces)

            remapped_landmarks = []

            for face in clipped_faces:
                # crop face
                (face_x, face_y, face_w, face_h) = face

                face_image = frame[int(face_y): int(face_y + face_h), int(face_x): int(face_x + face_w)]

                # detect landmarks
                face_landmarks = self.landmark_model.predict(face_image)

                # remap landmarks for original image
                remapped_face_landmarks = []

                for (x, y) in face_landmarks:
                    remapped_face_landmarks.append((face_x + x, face_y + y))

                remapped_landmarks.append(remapped_face_landmarks)

            ####################### draw results on the image #######################
            utils.draw_boxes(frame, clipped_faces)

            for face_remapped_landmarks in remapped_landmarks:
                utils.draw_landmarks(frame, face_remapped_landmarks)

            for face_five_landmarks in five_landmarks:
                utils.draw_landmarks(frame, face_five_landmarks, (0, 255, 0))
            #######################                           #######################

            cv2.imshow("image", frame)
            key = cv2.waitKey(1)

            if key == 27:
                print("processing stopped")
                return

    @staticmethod
    def close():
        print('close video processor')
