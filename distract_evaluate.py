import numpy as np
import cv2
import imutils
from keras.applications.mobilenet import preprocess_input
from keras.models import load_model
from keras.applications.mobilenet import MobileNet
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

model_path = '/Users/ranxin/Downloads/Distracted_mobilenet_full.h5'  # /home/hduser/Distracted_vgg16_full.h5
model = load_model(model_path)
# graph = tf.get_default_graph()
print(model.summary())

y_actual = []
y_predict = []

for i in range(10):
    files = os.listdir("/Users/ranxin/Downloads/test/c" + str(i))
    for file in files:
        if file[-3:] == 'jpg':
            print(file)
            frame = cv2.imread("/Users/ranxin/Downloads/test/c" + str(i) + "/" + file)
            image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
            image = image.reshape((-1, 224, 224, 3))
            image = preprocess_input(image)
            ynew = model.predict_classes(image)[0]
            y_actual.append(i)
            y_predict.append(ynew)

print(classification_report(y_actual, y_predict))
