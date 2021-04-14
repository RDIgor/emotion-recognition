import cv2
import argparse


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def crop_image(frame, box):
    (x, y, w, h) = box
    return frame[int(y): int(y + h), int(x): int(x + w)]


def load_image(file_name):
    return cv2.imread(file_name)


def show_image(window, image, delay=0):
    cv2.imshow(window, image)
    cv2.waitKey(delay)


def resize(image, width, height):
    return cv2.resize(image, (width, height))


def draw_bboxes(image, boxes):
    for (x, y, w, h, class_id, confidence) in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)


def draw_boxes(image, boxes, color=(0, 255, 0)):
    for (x, y, w, h) in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 4)


def draw_landmarks(image, landmarks, color=(0, 0, 255), thickness=3):
    for (landmark_x, landmark_y) in landmarks:
        cv2.circle(image, (int(landmark_x), int(landmark_y)), thickness, color, -1)


def draw_face_landmarks(image, list_dictionary):
    for dictionary in list_dictionary:
        for box, shape in dictionary.items():
            (x, y, w, h) = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)

            for (landmark_x, landmark_y) in shape:
                cv2.circle(image, (landmark_x, landmark_y), 3, (0, 0, 255), -1)


# 0 - x, 1 -y, w - 2, h - 3
def clip_rects(image, rects):
    result = []

    (image_h, image_w, channels) = image.shape

    for rect in rects:
        (x, y, w, h) = rect

        if x < 0:
            x = 0

        if y < 0:
            y = 0

        if x + w > image_w:
            w = image_h - x

        if y + h > image_h:
            h = image_h - y

        result.append([x, y, w, h])

    return result
