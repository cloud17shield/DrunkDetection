import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
import cv2
import imutils
from keras.applications.mobilenet import preprocess_input
from keras.models import load_model
from keras.applications.mobilenet import MobileNet

model_path = '/Users/lilingxiao/Documents/HKU/Project/Distracted_mobilenet_full_6c.h5'  # /home/hduser/Distracted_vgg16_full.h5
model = load_model(model_path)
# graph = tf.get_default_graph()
print(model.summary())
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
    image = image.reshape((-1, 224, 224, 3))
    image = preprocess_input(image)
    # global graph
    # global model
    # with graph.as_default():
    ynew = model.predict_classes(image)
    print("prediction", type(ynew), ynew)
    result_dic = {0: "normal driving", 1: "texting", 2: "talking on the phone",
                  3: "operating on the radio",
                  4: "drinking", 5: "reaching behind"}
    cv2.putText(frame, "status: " + result_dic[ynew[0]], (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
cap.stop()
