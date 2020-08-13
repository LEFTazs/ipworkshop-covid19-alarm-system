#!/usr/bin/env python

from flask import Flask, render_template, Response, jsonify
import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms


app = Flask(__name__)
model = None
data_transform = transforms.Compose([transforms.ToTensor(),])
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
statistic = {"mask": 0, "no-mask": 0, "incorrect-mask": 0}


@app.route('/')
def index():
    return render_template('index.html')


def gen(vc):
    while True:
        has_frame, frame = vc.read()
        if has_frame:
            prediction = get_prediction_from_model(frame)
            update_statistic(prediction[0])
            frame = draw_bboxes_on_image(frame, prediction[0])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(cv2.VideoCapture(0)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/rooms/1/statistics')
def statistics():
    return jsonify(statistic)


@torch.no_grad()
def get_prediction_from_model(img):
    img = data_transform(img)
    img = img.to(device)
    pred = model([img])
    return pred


def update_statistic(prediction):
    labels = prediction['labels']
    counts = torch.bincount(labels, minlength=3)
    statistic['no-mask'] = counts[0].item()
    statistic['mask'] = counts[1].item()
    statistic['incorrect-mask'] = counts[2].item()


def draw_bboxes_on_image(img, prediction):
    for box, label in zip(prediction['boxes'], prediction['labels']):
        xmin, ymin, xmax, ymax = box
        color = __choose_bbox_color(label)
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 4)
    return img


def __choose_bbox_color(label):
    if label == 0:
        return 0, 0, 255
    elif label == 1:
        return 0, 255, 0
    elif label == 2:
        return 255, 0, 0


def create_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_saved_model_state(path):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    model.to(device)


if __name__ == '__main__':
    model = create_model(3)
    absolute_path_to_saved_model = "change/this/path"
    load_saved_model_state(absolute_path_to_saved_model)

    app.run(host='0.0.0.0', debug=False)
