import numpy as np
import cv2


class FaceAligner:
    def __init__(self):
        pass

    @staticmethod
    def align(face_image, five_face_points):
        # extract left and right eye
        left_eye = five_face_points[0]
        right_eye = five_face_points[1]

        d_y = right_eye[1] - left_eye[1]
        d_x = right_eye[0] - left_eye[0]

        # compute angle
        angle = (np.arctan(d_y / d_x) * 180) / np.pi

        (h, w) = face_image.shape[:2]

        m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)

        rotated = cv2.warpAffine(face_image, m, (w, h))

        return rotated

