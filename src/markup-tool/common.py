import cv2
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QImage, QPixmap


def set_image(cv_image, label: QLabel):
    label_width = label.width()
    label_height = label.height()

    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    cv_image = cv2.resize(cv_image, (label_width, label_height))

    (h, w, ch) = cv_image.shape

    qt_image = QImage(cv_image.data, w, h, ch * w, QImage.Format_RGB888)

    label.setPixmap(QPixmap.fromImage(qt_image))