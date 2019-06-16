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
# import pydoop.hdfs as hdfs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from io import BytesIO
from PIL import Image

conf = SparkConf().setAppName("drunk video stream").setMaster("yarn")
sc = SparkContext(conf=conf)
ssc = StreamingContext(sc, 1)
sql_sc = SQLContext(sc)
input_topic = 'input'
output_topic = 'output'
brokers = "G01-01:2181,G01-02:2181,G01-03:2181,G01-04:2181,G01-05:2181,G01-06:2181,G01-07:2181,G01-08:2181," \
          "G01-09:2181,G01-10:2181,G01-11:2181,G01-12:2181,G01-13:2181,G01-14:2181,G01-15:2181,G01-16:2181 "


def my_decoder(s):
    if s is None:
        return None
    return s


kafkaStream = KafkaUtils.createStream(ssc, 'G01-01:2181', 'test-consumer-group', {input_topic: 1}, valueDecoder=my_decoder)
producer = KafkaProducer(bootstrap_servers='G01-01:9092',compression_type='gzip',batch_size=163840,buffer_memory=33554432,max_request_size=20485760)

csv_file_path = "file:///home/hduser/DrunkDetection/train_data48.csv"
predictor_path = "/home/hduser/DrunkDetection/shape_predictor_68_face_landmarks.dat"
model_path = "/home/hduser/DrunkDetection/rf48.pickle"

df = pd.read_csv(csv_file_path,  index_col=0)
print(df.columns)
df_y = df['label'] == 3
df_X = df[['x' + str(i) for i in range(1, 49)] + ['y' + str(j) for j in range(1, 49)]]
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=15)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
fa = FaceAligner(predictor, desiredFaceWidth=300)
with open(model_path, 'rb') as f:
    clf2 = pickle.load(f)


def handler(message):
    records = message.collect()
    for record in records:
        print('record', record, type(record))
        print('-----------')
        print('tuple', record[0], record[1], type(record[0]), type(record[1]))
        # producer.send(output_topic, b'message received')
        key = record[0]
        value = record[1]

        print("start processing")
        image = Image.frombytes('RGB', (385, 386), value, 'raw')
        # img = cv2.imread("/tmp/" + key)
        img = np.array(image, dtype=np.uint8)
        print(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        dic = {}
        x_values = [[] for _ in range(48)]
        y_values = [[] for _ in range(48)]

        for face in faces:
            (x, y, w, h) = rect_to_bb(face)
            # faceOrig = imutils.resize(img[y: y + h, x: x + w], width=300)
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
        # with open(model_path, 'rb') as f:
        #     clf2 = pickle.load(f)
        predict_value = 1 if True in clf2.predict(X_score) else 0
        cv2.putText(img, "Drunk: " + str(predict_value), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print("drunk prediction:", predict_value)
        producer.send(output_topic, value=img.tobytes())
        producer.flush()
        print("predict over")


kafkaStream.foreachRDD(handler)
ssc.start()
ssc.awaitTermination()
