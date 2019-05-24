from pyspark import SparkContext, SparkConf
import pickle
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import numpy as np
import imutils
import dlib
import cv2
import os
import pandas as pd
import pydoop.hdfs as hdfs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

conf = SparkConf().setAppName("drunk detection").setMaster("yarn")
sc = SparkContext(conf=conf)

csv_file_path = "hdfs:///drunkdetection/train_data48.csv"
predictor_path = "hdfs:///drunkdetection/shape_predictor_68_face_landmarks.dat"
image_path = "hdfs:///drunkdetection/drunk3.jpg"
model_path = "hdfs:///drunkdetection/rf48.pickle"

with hdfs.open("/drunkdetection/train_data48.csv") as csv:
    df = pd.read_csv(csv,  index_col=0)
print(df.columns)
df_y = df['label'] == 3
df_X = df[['x' + str(i) for i in range(1, 49)] + ['y' + str(j) for j in range(1,49)]]
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=15)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

detector = dlib.get_frontal_face_detector()
with hdfs.open("/drunkdetection/shape_predictor_68_face_landmarks.dat") as p:
    predictor = dlib.shape_predictor(p)
fa = FaceAligner(predictor, desiredFaceWidth=300)
with hdfs.open("/drunkdetection/drunk3.jpg") as f:
    img = cv2.imread(f)
print(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector(gray, 1)

dic = {}
x_values = [[] for _ in range(48)]
y_values = [[] for _ in range(48)]

for face in faces:
    (x, y, w, h) = rect_to_bb(face)
    faceOrig = imutils.resize(img[y: y + h, x: x + w], width=300)
    faceAligned = fa.align(img, gray, face)

    dets = detector(faceAligned, 1)
    num_face = len(dets)
    print("num of face:", num_face)
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
with hdfs.open("/drunkdetection/rf48.pickle", 'rb') as f:
    clf2 = pickle.load(f)
    print("drunk prediction:", clf2.predict(X_score))
