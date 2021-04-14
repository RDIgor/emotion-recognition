from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QWidget,
    QComboBox)

from PyQt5.QtGui import (
    QIcon,
    QPixmap,
    QImage)

from PyQt5.QtCore import Qt

import cv2
import os
from PyQt5.QtCore import QTimer
from src.utils import crop_image, draw_landmarks
from src.image_processor import ImageProcessor
from src.common import Face
from common import set_image


class MarkupWidget(QWidget):
    def __init__(self, parent=None, args=None):
        super(MarkupWidget, self).__init__(parent)
        self.image_processor = ImageProcessor(args=args, config=None)
        self.video_capture = cv2.VideoCapture()
        self.current_image = None
        self.approved_data = []
        self.current_faces = []  # faces
        self.current_face_index = -1
        self.global_data_index = 0
        self.dst_data_folder = './dataset'
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.next_frame)

        if not os.path.exists(self.dst_data_folder):
            os.mkdir(self.dst_data_folder)

        # create left video(image) layout
        left_vertical_layout = QVBoxLayout()
        right_vertical_layout = QVBoxLayout()
        face_bottom_layout = QHBoxLayout()
        source_buttons_layout = QHBoxLayout()
        faces_layout = QHBoxLayout()
        top_main_layout = QHBoxLayout()
        main_layout = QVBoxLayout()

        # create source label
        self.source_label = QLabel()
        self.source_label.setMinimumSize(800, 800)
        self.source_label.setMaximumSize(800, 800)
        self.source_label.setStyleSheet('border: 1px solid red')

        # create face label
        self.face_label = QLabel()
        self.face_landmarks_label = QLabel()
        self.approved_label = QLabel()
        self.approved_label.setMaximumSize(1000, 30)
        self.approved_label.setMinimumHeight(30)
        self.approved_label.setStyleSheet('background-color: red')
        self.face_label.setMinimumHeight(500)
        self.face_label.setMaximumHeight(500)
        self.face_landmarks_label.setMaximumHeight(500)
        self.face_landmarks_label.setMaximumHeight(500)
        self.face_label.setStyleSheet('border: 1px solid red')
        self.face_landmarks_label.setStyleSheet('border: 1px solid red')

        # create buttons for faces and combo box
        self.previous_face_button = QPushButton('<')
        self.next_face_button = QPushButton('>')
        self.approve_face_button = QPushButton()
        self.cancel_face_button = QPushButton()
        self.classifications = QComboBox()
        self.classifications.setMaximumSize(120, 30)
        self.approve_face_button.setMaximumSize(60, 30)
        self.approve_face_button.setMinimumSize(60, 30)
        self.approve_face_button.setIcon(self.load_icon('./icons/approve.png'))
        self.approve_face_button.clicked.connect(self.approve_face)
        self.cancel_face_button.setIcon(self.load_icon('./icons/cancel.png'))
        self.cancel_face_button.setMaximumSize(60, 30)
        self.approve_face_button.setMaximumSize(60, 30)
        self.cancel_face_button.setMinimumSize(60, 30)
        self.approve_face_button.setMinimumSize(60, 30)
        self.cancel_face_button.clicked.connect(self.cancel_face)
        self.previous_face_button.clicked.connect(self.previous_face)
        self.next_face_button.clicked.connect(self.next_face)
        self.previous_face_button.setMaximumSize(60, 30)
        self.next_face_button.setMaximumSize(60, 30)
        self.previous_face_button.setMinimumSize(60, 30)
        self.next_face_button.setMinimumSize(60, 30)
        self.classifications.addItems(['smile', 'not a smile'])

        # create source buttons
        self.previous_button = QPushButton('<')
        self.next_button = QPushButton('>')
        self.play_button = QPushButton('play')
        self.stop_button = QPushButton('stop')
        self.previous_button.setMaximumSize(100, 100)
        self.next_button.setMaximumSize(100, 100)
        self.play_button.setMaximumSize(100, 100)
        self.stop_button.setMaximumSize(100, 100)
        self.previous_button.setMinimumSize(100, 35)
        self.next_button.setMinimumSize(100, 35)
        self.play_button.setMinimumSize(100, 35)
        self.stop_button.setMinimumSize(100, 35)
        self.previous_button.clicked.connect(self.previous_frame)
        self.next_button.clicked.connect(self.next_frame)
        self.play_button.clicked.connect(self.play)
        self.stop_button.clicked.connect(self.stop)

        # form layouts
        left_vertical_layout.addWidget(self.source_label)
        left_vertical_layout.addStretch()

        face_bottom_layout.addWidget(self.previous_face_button)
        face_bottom_layout.addStretch()
        face_bottom_layout.addWidget(self.classifications)
        face_bottom_layout.addSpacing(5)
        face_bottom_layout.addWidget(self.approve_face_button)
        face_bottom_layout.addSpacing(5)
        face_bottom_layout.addWidget(self.cancel_face_button)
        face_bottom_layout.addStretch()
        face_bottom_layout.addWidget(self.next_face_button)

        faces_layout.addWidget(self.face_label)
        faces_layout.addSpacing(20)
        faces_layout.addWidget(self.face_landmarks_label)

        right_vertical_layout.addLayout(faces_layout)
        right_vertical_layout.addSpacing(10)
        right_vertical_layout.addWidget(self.approved_label)
        right_vertical_layout.addSpacing(30)
        right_vertical_layout.addLayout(face_bottom_layout)
        right_vertical_layout.addStretch()

        source_buttons_layout.addStretch()
        source_buttons_layout.addWidget(self.previous_button)
        source_buttons_layout.addSpacing(10)
        source_buttons_layout.addWidget(self.play_button)
        source_buttons_layout.addWidget(self.stop_button)
        source_buttons_layout.addSpacing(10)
        source_buttons_layout.addWidget(self.next_button)
        source_buttons_layout.addStretch()

        top_main_layout.addLayout(left_vertical_layout)
        top_main_layout.addSpacing(20)
        top_main_layout.addLayout(right_vertical_layout)

        main_layout.addLayout(top_main_layout)
        main_layout.addSpacing(20)
        main_layout.addLayout(source_buttons_layout)

        self.setFocusPolicy(Qt.StrongFocus)
        self.setLayout(main_layout)

    @staticmethod
    def load_icon(path: str):
        icon = QIcon()
        icon.addPixmap(QPixmap(path), QIcon.Disabled)

        return icon

    def set_source(self, path: str):
        if path:
            extension = path.split('.')[-1]

            if extension in 'mp4, avi, mov':  # video
                self.change_source_buttons_state(True)

                self.video_capture.open(path)

                if self.video_capture.isOpened():
                    self.next_frame()

                else:
                    print("Error opening the video, path {}".format(path))
            else:
                self.change_source_buttons_state(False)

                image = cv2.imread(path)

                if image is not None:
                    set_image(image, self.source_label)
        else:
            print('Invalid source')

    def change_source_buttons_state(self, state: bool):
        self.next_button.setEnabled(state)
        self.previous_button.setEnabled(state)
        self.play_button.setEnabled(state)
        self.stop_button.setEnabled(state)

    def get_next_frame(self):
        ret, frame = self.video_capture.read()

        if not ret:
            return None
        else:
            return frame

    def save_data(self, original_image, landmarks: list, label: int):
        h, w = original_image.shape[:2]

        cv2.imwrite(self.dst_data_folder + "/{}.png".format(self.global_data_index), original_image)

        with open(self.dst_data_folder + "/{}.txt".format(self.global_data_index), 'w') as f:
            for (x, y) in landmarks:
                f.write("{:.2f},{:.2f}\n".format(float(x) / w, float(y) / h))

            f.write(str(label))

        self.global_data_index += 1

    def save_data_batch(self, images: list):
        for index, image, landmarks, label in images:
            self.save_data(image, landmarks, label)

    def next_frame(self):
        # save data
        if self.approved_data:
            self.save_data_batch(self.approved_data)

        # reset previous state
        self.approved_data = []
        self.current_faces = []
        self.current_face_index = -1

        # read next frame
        frame = self.get_next_frame()

        self.current_image = frame

        if frame is not None:
            # detect faces and landmarks
            self.detect_faces(frame)

            # add faces and landmarks
            if self.current_faces:
                # set face image on label
                self.next_face()

            # set source image
            set_image(frame, self.source_label)

    def previous_frame(self):
        pos = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)

        if pos < 1:
            return

        pos -= 2

        if pos > 0:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, pos)

            self.next_frame()

    def next_face(self):
        if self.current_face_index >= len(self.current_faces) - 1:
            return

        self.current_face_index += 1

        if self.current_face_index >= 0:
            face = self.current_faces[self.current_face_index]

            if self.was_approved(self.current_face_index):
                self.approved_label.setStyleSheet('background-color: green')
            else:
                self.approved_label.setStyleSheet('background-color: red')

            # set image on the face label
            set_image(crop_image(self.current_image, face.box), self.face_label)

            # draw landmarks
            face_landmarks_image = crop_image(self.current_image, face.box).copy()
            draw_landmarks(face_landmarks_image, face.landmarks, thickness=1)

            # set image on the face landmarks label
            set_image(face_landmarks_image, self.face_landmarks_label)

    def previous_face(self):
        if self.current_face_index < 1:
            return
        else:
            self.current_face_index -= 2

        self.next_face()

    def approve_face(self):
        self.approved_label.setStyleSheet('background-color: green')

        if 0 <= self.current_face_index < len(self.current_faces):
            face = self.current_faces[self.current_face_index]
            classification = self.classifications.currentIndex()  # 0 - smile, 1 - not a smile

            self.approved_data.append((
                self.current_face_index,
                crop_image(self.current_image, face.box),
                face.landmarks, classification))

    def was_approved(self, current_index):
        for index, image, landmarks, classification in self.approved_data:
            if index == current_index:
                return True

        return False

    def cancel_face(self):
        print(self.approved_data)
        self.approved_label.setStyleSheet('background-color: red')

        if 0 <= self.current_face_index < len(self.current_faces):
            for i in range(0, len(self.approved_data)):
                approved_index = self.approved_data[i][0]

                if approved_index == self.current_face_index:
                    self.approved_data.pop(i)
                    break

        print(self.approved_data)

    def detect_faces(self, frame):
        if frame is not None:
            # detect faces
            try:
                self.current_faces = self.image_processor.process(frame)
            except Exception as err:
                print(err)

    def play(self):
        self.play_button.setEnabled(False)

        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        ms = int(1000 / float(fps))

        self.video_timer.start(ms)

    def stop(self):
        self.play_button.setEnabled(True)
        self.video_timer.stop()

    def keyPressEvent(self, a0) -> None:
        key = a0.key()
        if key == Qt.Key_Left:
            self.previous_frame()
        elif key == Qt.Key_Right:
            self.next_frame()
