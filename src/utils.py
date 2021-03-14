import cv2


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


def draw_boxes(image, boxes):
    for (x, y, w, h) in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)


def draw_landmarks(image, landmarks):
    for face_landmarks in landmarks:
        for (landmark_x, landmark_y) in face_landmarks:
            cv2.circle(image, (landmark_x, landmark_y), 3, (0, 0, 255), -1)


def draw_face_landmarks(image, list_dictionary):
    for dictionary in list_dictionary:
        for box, shape in dictionary.items():
            (x, y, w, h) = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)

            for (landmark_x, landmark_y) in shape:
                cv2.circle(image, (landmark_x, landmark_y), 3, (0, 0, 255), -1)