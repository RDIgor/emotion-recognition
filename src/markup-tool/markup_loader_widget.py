import os

import cv2
import ntpath
from os import listdir
from os.path import isfile, join
from common import set_image
from src.common import Emotion
from src.utils import draw_landmarks

from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QWidget)

from PyQt5.QtCore import Qt


class MarkupLoaderWidget(QWidget):
    def __init__(self, parent=None):
        super(MarkupLoaderWidget, self).__init__(parent)
        self.images = []
        self.markup = []
        self.index = -1

        faces_layout = QHBoxLayout()
        vertical_info_layout = QVBoxLayout()
        buttons_layout = QHBoxLayout()
        main_layout = QVBoxLayout()

        self.face_label = QLabel()
        self.face_landmarks_label = QLabel()
        self.id_label = QLabel()
        self.classification_label = QLabel()
        self.face_label.setStyleSheet('border: 1px solid red')
        self.face_label.setMaximumSize(400, 500)
        self.face_label.setMinimumSize(400, 500)
        self.face_landmarks_label.setStyleSheet('border: 1px solid red')
        self.face_landmarks_label.setMinimumSize(400, 500)
        self.face_landmarks_label.setMaximumSize(400, 500)
        self.id_label.setAlignment(Qt.AlignCenter)
        self.id_label.setStyleSheet('border: 1px solid red')
        self.id_label.setMinimumSize(200, 50)
        self.id_label.setMaximumSize(200, 50)
        self.classification_label.setAlignment(Qt.AlignCenter)
        self.classification_label.setStyleSheet('border: 1px solid red')
        self.classification_label.setMinimumSize(200, 50)
        self.classification_label.setMaximumSize(200, 50)

        self.previous_button = QPushButton('<')
        self.next_button = QPushButton('>')
        self.remove_button = QPushButton('Remove')
        self.previous_button.clicked.connect(self.previous_data)
        self.next_button.clicked.connect(self.next_data)
        self.remove_button.clicked.connect(self.remove_data)
        self.previous_button.setMaximumSize(100, 40)
        self.previous_button.setStyleSheet('background-color: #90EE90')
        self.next_button.setMaximumSize(100, 40)
        self.next_button.setStyleSheet('background-color: #90EE90')
        self.remove_button.setMaximumSize(100, 40)
        self.remove_button.setStyleSheet('background-color: red')

        vertical_info_layout.addWidget(self.classification_label)
        vertical_info_layout.addSpacing(20)
        vertical_info_layout.addWidget(self.id_label)

        faces_layout.addStretch()
        faces_layout.addWidget(self.face_label)
        faces_layout.addSpacing(10)
        faces_layout.addLayout(vertical_info_layout)
        faces_layout.addSpacing(10)
        faces_layout.addWidget(self.face_landmarks_label)
        faces_layout.addStretch()

        buttons_layout.addWidget(self.previous_button)
        buttons_layout.addSpacing(20)
        buttons_layout.addWidget(self.remove_button)
        buttons_layout.addSpacing(20)
        buttons_layout.addWidget(self.next_button)

        main_layout.addLayout(faces_layout)
        main_layout.addLayout(buttons_layout)

        self.setLayout(main_layout)

    def set_markup_folder(self, path: str):
        self.setFocusPolicy(Qt.StrongFocus)

        self.markup = []
        self.images = []
        self.index = -1

        files = listdir(path)
        files.sort()
        for f in files:
            file = join(path, f)
            if isfile(file):
                extension = file.split('.')[-1]
                if extension in ('png', 'jpg', 'jpeg'):
                    self.images.append(file)
                elif extension == 'txt':
                    self.markup.append(file)

        self.next_data()

    def next_data(self):
        if self.index >= len(self.markup) - 1:
            return

        self.index += 1

        if self.index >= 0:
            image = cv2.imread(self.images[self.index])

            if image is not None:
                landmarks = []
                label = None
                path = self.markup[self.index]
                filename = ntpath.basename(path)
                with open(path, 'r') as reader:
                    for line in reader:
                        if ',' in line:
                            values = line.split(',')
                            x = float(values[0])
                            y = float(values[1])

                            landmarks.append((x, y))
                        else:
                            label = int(line)
                            break

                if landmarks and label is not None:
                    self.show_sample(image, landmarks, label, filename)

    def previous_data(self):
        if self.index < 1:
            return

        self.index -= 2

        self.next_data()

    def show_sample(self, image, landmarks: list, classification: int, filename: str):
        # set image on the face label
        set_image(image, self.face_label)

        # prepare landmarks
        landmarks_image = cv2.resize(image, (self.face_landmarks_label.width(), self.face_landmarks_label.height()))
        h, w = landmarks_image.shape[:2]
        absolute_landmarks = []
        for (x, y) in landmarks:
            absolute_landmarks.append((int(x * w), int(y * h)))

        draw_landmarks(landmarks_image, absolute_landmarks, (0, 0, 255), 2)

        # set image on the face landmarks label
        set_image(landmarks_image, self.face_landmarks_label)

        # set classification label
        emotion = Emotion(classification)

        if emotion == Emotion.SMILE:
            self.classification_label.setStyleSheet('background-color: green')
            self.classification_label.setText('smile')
        else:
            self.classification_label.setStyleSheet('background-color: red')
            self.classification_label.setText('not a smile')

        # set filename
        self.id_label.setText(filename)

    def keyPressEvent(self, a0) -> None:
        key = a0.key()
        if key == Qt.Key_Left:
            self.previous_data()
        elif key == Qt.Key_Right:
            self.next_data()

    def remove_data(self):
        markup_path = self.markup[self.index]
        image_path = self.images[self.index]

        os.remove(markup_path)
        os.remove(image_path)

        del self.markup[self.index]
        del self.images[self.index]

        self.previous_data()
