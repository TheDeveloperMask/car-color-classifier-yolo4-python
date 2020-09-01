# car-color-classifier-yolo4-python
Car color recognition example with YOLOv4 object detector

## Introduction

A Python example for using [Spectrico's car color classifier](http://spectrico.com/car-color-recognition.html). It consists of an object detector for finding the cars, and a classifier to recognize the colors of the detected cars. The object detector is an implementation of YOLOv4 (OpenCV DNN backend). YOLOv4 weights were downloaded from [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights). The classifier is based on MobileNet v3 (TensorFlow backend).

---
## Object Detection and Classification in images
This example takes an image as input, detects the cars using YOLOv4 object detector, crops the car images, resizes them to the input size of the classifier, and recognizes the color of each car. The result is shown on the display and saved as output.jpg image file.


#### Usage
Use --help to see usage of car_color_classifier_yolo4.py:
```
$ python car_color_classifier_yolo4.py --image cars.jpg
```
```
$ python car_color_classifier_yolo4.py [-h] [--yolo MODEL_PATH] [--confidence CONFIDENCE] [--threshold THRESHOLD] [--image]

required arguments:
  -i, --image              path to input image

optional arguments:
  -h, --help               show this help message and exit
  -y, --yolo MODEL_PATH    path to YOLO model weight file, default yolo-coco
  --confidence CONFIDENCE  minimum probability to filter weak detections, default 0.5
  --threshold THRESHOLD    threshold when applying non-maxima suppression, default 0.3
```
![image](https://github.com/spectrico/car-color-classifier-yolo4-python/blob/master/output.jpg?raw=true)

---
## Requirements
  - python
  - numpy
  - tensorflow
  - opencv
  - yolov4.weights must be downloaded from [https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) and saved in folder yolov4

---
## Configuration

The settings are stored in python file named config.py:
```
model_file = "model-weights-spectrico-car-colors-recognition-mobilenet_v3-224x224-180420.pb"
label_file = "labels.txt"
input_layer = "input_1"
output_layer = "Predictions/Softmax/Softmax"
classifier_input_size = (224, 224)
```
***model_file*** is the path to the car color classifier
***classifier_input_size*** is the input size of the classifier
***label_file*** is the path to the text file, containing a list with the supported colors

---
## Credits
The examples are based on the tutorial by Adrian Rosebrock: [YOLO object detection with OpenCV](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)

The YOLOv4 object detector is from: [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
```
@article{bochkovskiy2020yolov4,
  title={YOLOv4: Optimal Speed and Accuracy of Object Detection},
  author={Bochkovskiy, Alexey and Wang, Chien-Yao and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2004.10934},
  year={2020}
}
```
The car color classifier is based on MobileNetV3 mobile architecture: [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
```
@inproceedings{howard2019searching,
  title={Searching for mobilenetv3},
  author={Howard, Andrew and Sandler, Mark and Chu, Grace and Chen, Liang-Chieh and Chen, Bo and Tan, Mingxing and Wang, Weijun and Zhu, Yukun and Pang, Ruoming and Vasudevan, Vijay and others},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={1314--1324},
  year={2019}
}
```
