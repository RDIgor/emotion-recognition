import cv2


def load_image(file_name):
    return cv2.imread(file_name)


def show_image(window, image):
    cv2.imshow(window, image)
    cv2.waitKey(0)


def resize(image, width, height):
    return cv2.resize(image, (width, height))


def draw_boxes(image, boxes):
    for (x, y, w, h, class_id, confidence) in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
