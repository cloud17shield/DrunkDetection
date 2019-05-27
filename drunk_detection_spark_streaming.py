import sys
from kafka import KafkaProducer, KafkaConsumer, TopicPartition
from kafka.errors import KafkaError, KafkaTimeoutError

from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
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
from sklearn.preprocessing import StandardScaler

conf = SparkConf().setAppName("drunkdetection").setMaster("yarn")
sc = SparkContext(conf=conf)
ssc = StreamingContext(sc, 3)
sql_sc = SQLContext(sc)
input_topic = 'input1'
output_topic = 'output'
brokers = "student49-x1:2181,student49-x2:2181,student50-x1:2181,student50-x2:2181"

kafkaStream = KafkaUtils.createStream(ssc, 'student49-x1:2181', 'test-consumer-group', {input_topic: 1})
producer = KafkaProducer(bootstrap_servers='student49-x1:9092')

with hdfs.open("/drunkdetection/train_data48.csv") as csv:
    df = pd.read_csv(csv,  index_col=0)
print(df.columns)
df_y = df['label'] == 3
df_X = df[['x' + str(i) for i in range(1, 49)] + ['y' + str(j) for j in range(1,49)]]
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=15)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
detector = dlib.get_frontal_face_detector()
hdfs.get("/drunkdetection/shape_predictor_68_face_landmarks.dat", "tmp/shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor("tmp/shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=300)


def handler(message):
    records = message.collect()
    for record in records:
        print('record', record, type(record))
        print('-----------')
        print('tuple', record[0], record[1], type(record[0]), type(record[1]))
        # producer.send(output_topic, b'message received')
        key = record[0]
        value = record[1]
        if len(key) > 10:
            image_path = value
            hdfs.get(image_path, "/tmp/" + key)
            img = cv2.imread("/tmp/" + key)
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
                predict_value = clf2.predict(X_score)
                print("drunk prediction:", predict_value)
                producer.send(output_topic, key=str(key).encode('utf-8'), value=str(predict_value).encode('utf-8'))
                producer.flush()
                print("predict over")


kafkaStream.foreachRDD(handler)
