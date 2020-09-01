# Copyright © 2020 by Spectrico
# Licensed under the MIT License

model_file = "model-weights-spectrico-car-colors-recognition-mobilenet_v3-224x224-180420.pb"  # path to the car color classifier
label_file = "labels.txt"   # path to the text file, containing list with the supported makes and models
input_layer = "input_1"
output_layer = "Predictions/Softmax/Softmax"
classifier_input_size = (224, 224) # input size of the classifier
