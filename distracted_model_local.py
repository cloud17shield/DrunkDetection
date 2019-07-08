import numpy as np
import cv2
import imutils
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
import tensorflow as tf

model_path = 'Distracted_vgg16_full.h5'  # /home/hduser/Distracted_vgg16_full.h5
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
    result_dic = {0: "normal driving", 1: "texting - right", 2: "talking on the phone - right",
                  3: "texting - left", 4: "talking on the phone - left", 5: "operating on the radio",
                  6: "drinking", 7: "reaching behind", 8: "hair and makeup", 9: "talking to passenger"}
    cv2.putText(frame, "status: " + result_dic[ynew[0]], (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
cap.stop()
