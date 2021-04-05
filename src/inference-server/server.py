import argparse
import json
import cv2
import base64
import numpy as np
from src.face_detector import FaceDetector
from src.face_processor import FaceProcessor
from flask import Flask, request, jsonify

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=True, help="configuration path")
args = vars(ap.parse_args())

with open(args["config"], 'r') as config:
    json_config = json.load(config)
    face_detector = FaceDetector(json_config)  # face detection
    face_processor = FaceProcessor(json_config)  # landmarks

app = Flask(__name__)

import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


@app.route('/faces', methods=['GET'])
def faces():
    json_data = request.get_json()

    # extract frame
    jpeg = json_data['frame']
    frame = cv2.imdecode(np.frombuffer(base64.b64decode(jpeg), dtype=np.uint8), flags=1)

    # detect faces and landmarks
    (face_boxes, five_landmarks) = face_detector.detect(frame)

    # generate data
    data = {'faces': face_boxes, 'five_landmarks': five_landmarks.tolist()}

    return jsonify(data)


@app.route('/landmarks', methods=['GET'])
def landmarks():
    json_data = request.get_json()

    # extract image
    jpeg = json_data['frame']
    face_image = cv2.imdecode(np.frombuffer(base64.b64decode(jpeg), dtype=np.uint8), flags=1)

    # extract face_box
    face_box = json_data['face_box']

    # extract five landmarks
    face_five_landmarks = json_data['face_five_landmarks']

    # extract landmarks and generate data
    data = {'landmarks': face_processor.process(face_image, face_box, face_five_landmarks).tolist()}

    return jsonify(data)


app.run()
