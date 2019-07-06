import pickle
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import numpy as np
import imutils
import dlib
import cv2
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

csv_file_path = "/Users/ranxin/PycharmProjects/DrunkDetection/train_data48-100.csv"
predictor_path = "/Users/ranxin/PycharmProjects/DrunkDetection/shape_predictor_68_face_landmarks.dat"
model_path = "/Users/ranxin/PycharmProjects/DrunkDetection/rf48-100.pickle"

df = pd.read_csv(csv_file_path, index_col=0)
print(df.columns)
df_y = df['label'] == 3
df_X = df[['x' + str(i) for i in range(1, 49)] + ['y' + str(j) for j in range(1, 49)]]
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=15)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
fa = FaceAligner(predictor, desiredFaceWidth=100)
with open(model_path, 'rb') as f:
    clf2 = pickle.load(f)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    if len(faces) >= 1:
        predict_value = 0

        for face in faces:
            dic = {}
            x_values = [[] for _ in range(48)]
            y_values = [[] for _ in range(48)]
            (x, y, w, h) = rect_to_bb(face)
            # faceOrig = imutils.resize(img[y: y + h, x: x + w], width=100)
            faceAligned = fa.align(frame, gray, face)

            dets = detector(faceAligned, 0)
            num_face = len(dets)
            print(num_face)
            if num_face == 1:
                for k, d in enumerate(dets):
                    shape = predictor(faceAligned, d)
                    for j in range(48):
                        x_values[j].append(shape.part(j).x)
                        y_values[j].append(shape.part(j).y)
                for i in range(48):
                    dic['x' + str(i + 1)] = x_values[i]
                    dic['y' + str(i + 1)] = y_values[i]

                df_score = pd.DataFrame(data=dic)
                df_score = df_score[['x' + str(i) for i in range(1, 49)] + ['y' + str(j) for j in range(1, 49)]]
                X_score = scaler.transform(df_score)
                if True in clf2.predict(X_score):
                    predict_value = 1
                    break
        cv2.putText(frame, "Drunk: " + str(predict_value), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # print("drunk prediction:", predict_value)
            # print("predict over")
            # print('send over!')

    else:
        current = int(time.time() * 1000)
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # print('send over!')
    # time.sleep(0.1)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
cap.stop()
