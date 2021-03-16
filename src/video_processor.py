import cv2
import json
import utils
import numpy as np
from face_processor import FaceProcessor
from face_detector import FaceDetector


class VideoProcessor:
    def __init__(self, args):
        # open config
        with open(args["config"]) as json_file:
            self.json_config = json.load(json_file)

        self.face_detector = FaceDetector()

        self.face_processor = FaceProcessor(self.json_config)

        # open video file
        self.capture = cv2.VideoCapture(args['input'])

        # open video writer
        # self.video_writer = VideoWriter("output.avi", 640, 640)

        if not self.capture.isOpened():
            print('Video file does not open')
        else:
            self.window_name = "result"

            cv2.namedWindow(self.window_name)
            cv2.createTrackbar("", self.window_name, 0, int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)), self.on_trackbar)

    def on_trackbar(self, val):
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, val)

    def start(self):
        while True:
            # read frame
            ret, frame = self.capture.read()

            if not ret:
                print('Failed to read frame')

                return

            (faces, five_landmarks) = self.face_detector.detect(frame)

            remapped_landmarks = []

            for i in range(0, len(faces)):
                face = faces[i]
                face_five_landmarks = five_landmarks[i]

                remapped_landmarks.append(self.face_processor.process(frame, face, face_five_landmarks))

            self.draw_debug(frame, faces, five_landmarks, remapped_landmarks)

            self.show(self.window_name, frame)

    @staticmethod
    def close():
        print('close video processor')

    @staticmethod
    def draw_debug(frame, faces, five_landmarks, face_landmarks):
        ####################### draw results on the image #######################
        utils.draw_boxes(frame, faces)

        for face_landmarks in face_landmarks:
            utils.draw_landmarks(frame, face_landmarks)

        for face_five_landmarks in five_landmarks:
            utils.draw_landmarks(frame, face_five_landmarks, (0, 255, 0))
        #######################                           #######################

    @staticmethod
    def show(window_name, frame):
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1)

        if key == 27:
            print("processing stopped")
            exit(1)