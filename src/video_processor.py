import cv2
import json
from video_writer import VideoWriter
from src.models.face_landmark_prediction_model import FaceLandmarkPredictionModel
from src.models.face_detection_model import FaceDetectionModel
from src.models.dlib_face_detection_model import DlibFaceDetectionModel
from src import utils


class VideoProcessor:
    def __init__(self, args):
        # open config
        with open(args["config"]) as json_file:
            self.json_config = json.load(json_file)

        # load face detection model
        self.face_model = DlibFaceDetectionModel(
            self.json_config["face_detection"]["weights"])

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
            faces = self.face_model.predict(frame)

            # detect landmarks
            landmarks = self.landmark_model.predict(frame, faces)

            # utils.draw_face_landmarks(frame, detections)
            utils.draw_boxes(frame, faces)
            utils.draw_landmarks(frame, landmarks)

            cv2.imshow("image", frame)
            key = cv2.waitKey(1)

    @staticmethod
    def close():
        print('close video processor')
