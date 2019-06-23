import pickle
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import numpy as np
import imutils
import dlib
import cv2
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

input_topic = 'input'
output_topic = 'output'
brokers = "G01-01:2181,G01-02:2181,G01-03:2181,G01-04:2181,G01-05:2181,G01-06:2181,G01-07:2181,G01-08:2181," \
          "G01-09:2181,G01-10:2181,G01-11:2181,G01-12:2181,G01-13:2181,G01-14:2181,G01-15:2181,G01-16:2181 "

csv_file_path = "file:///home/hduser/DrunkDetection/train_data48.csv"
predictor_path = "/home/hduser/DrunkDetection/shape_predictor_68_face_landmarks.dat"
model_path = "/home/hduser/DrunkDetection/rf48.pickle"

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


def handler(raw):
    val = raw.get(0)
    print("row type:", type(val))
    records = raw.collect()
    for record in records:
        try:
            print('record', len(record), type(record))
            print('-----------')
            print('tuple', type(record[0]), type(record[1]))
        except Exception:
            print("error")
        # producer.send(output_topic, b'message received')
        key = record[0]
        value = record[1]

        print("len", len(key), len(value))

        print("start processing")
        image = np.asarray(bytearray(value), dtype="uint8")
        # image = np.frombuffer(value, dtype=np.uint8)
        # img = image.reshape(300, 400, 3)
        # img = cv2.imread("/tmp/" + key)
        img = cv2.imdecode(image, cv2.IMREAD_ANYCOLOR)
        print('img shape', img, img.shape)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        if len(faces) >= 1:
            predict_value = 0

            for face in faces:
                dic = {}
                x_values = [[] for _ in range(48)]
                y_values = [[] for _ in range(48)]
                (x, y, w, h) = rect_to_bb(face)
                # faceOrig = imutils.resize(img[y: y + h, x: x + w], width=100)
                faceAligned = fa.align(img, gray, face)

                dets = detector(faceAligned, 1)
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
            cv2.putText(img, "Drunk: " + str(predict_value), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print("drunk prediction:", predict_value)
            print("predict over")

        else:
            cv2.putText(img, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # producer.send(output_topic, value=cv2.imencode('.jpg', img)[1].tobytes(), key=key.encode('utf-8'))
        # producer.flush()
        print('send over!')
    pass


# Spark session
spark = SparkSession \
    .builder \
    .appName("drunk streaming structure") \
    .getOrCreate()

# Subscribe to 1 topic, read from kafka, length("value").alias("len"), "timestamp"
df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "G01-01:9092") \
    .option("subscribe", 'input') \
    .option("startingOffsets", "latest") \
    .load() \
    .select(decode("key", 'UTF-8').alias("key"), "value")

# process
df.printSchema()
df_sorted = df.sort(col("key").asc())
df_sorted.show(10)
for row in df_sorted.rdd.collect():
    try:
        print('record', len(row), type(row))
        print('-----------')
        print('tuple', type(row[0]), type(row[1]))
    except Exception:
        print("error")
    # producer.send(output_topic, b'message received')
    key = row[0]
    value = row[1]

    print("len", len(key), len(value))

    print("start processing")

# Write key-value data from a DataFrame to a specific Kafka topic specified in an option
ds = df_sorted \
    .writeStream \
    .format("console") \
    .trigger(continuous='1 second') \
    .outputMode('update') \
    .start()

ds.awaitTermination()
