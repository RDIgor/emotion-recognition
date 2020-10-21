import argparse
import json
import os
from src.models.yolo import YoloModel
from src import utils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, help="path to input image")
ap.add_argument("-c", "--config", required=True, help="configuration path")

args = vars(ap.parse_args())

if __name__ == '__main__':
    config_path = args['config']

    if not os.path.exists(config_path):
        exit(1)

    with open(config_path) as json_file:
        config = json.load(json_file)

    yolo_model = YoloModel(config["detection"]["config"], config["detection"]["weights"])

    image = utils.load_image(args['image'])
    image = utils.resize(image, 1024, 768)

    boxes = yolo_model.predict(image)

    utils.draw_boxes(image, boxes)
    utils.show_image('test', image)

