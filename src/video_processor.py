import cv2
import json
import utils
import time
from image_processor import ImageProcessor


class VideoProcessor:
    def __init__(self, args):
        # open config
        with open(args.config) as json_file:
            json_config = json.load(json_file)

        # open video file
        self.capture = cv2.VideoCapture(args.input)

        # open video writer
        # self.video_writer = VideoWriter("output.avi", 640, 640)

        if not self.capture.isOpened():
            print('Video file does not open')
        else:
            self.image_processor = ImageProcessor(json_config, args)
            self.window_name = "result"

            cv2.namedWindow(self.window_name)
            cv2.createTrackbar("", self.window_name, 0, int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                               self.on_trackbar)

    def on_trackbar(self, val):
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, val)

    def start(self):
        while True:
            # read frame
            ret, frame = self.capture.read()

            if not ret:
                print('Failed to read frame')

                return

            begin = time.time()

            # process frame. Detects faces and extracts landmarks
            faces = self.image_processor.process(frame)

            # compute fps
            fps = 1.0 / (time.time() - begin)

            # draw results
            self.draw_debug(frame, faces, fps)

            # show
            self.show(self.window_name, frame)

    @staticmethod
    def close():
        print('close video processor')

    @staticmethod
    def draw_debug(frame, faces, fps):
        for face in faces:
            utils.draw_boxes(frame, [face.box], (0, 255, 0) if face.applicability else (0, 0, 255))
            utils.draw_landmarks(frame, face.five_landmarks, (0, 255, 0))
            utils.draw_landmarks(utils.crop_image(frame, face.box), face.landmarks)

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 4)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 4)

    @staticmethod
    def show(window_name, frame):
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1)

        if key == 27:
            print("processing stopped")
            exit(1)
        elif key == 32:  # space
            cv2.waitKey(0)
