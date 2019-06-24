import pickle
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import numpy as np
import imutils
import dlib
import cv2
import os
from kafka import KafkaProducer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

input_topic = 'input'
output_topic = 'output'
brokers = "G01-01:2181,G01-02:2181,G01-03:2181,G01-04:2181,G01-05:2181,G01-06:2181,G01-07:2181,G01-08:2181," \
          "G01-09:2181,G01-10:2181,G01-11:2181,G01-12:2181,G01-13:2181,G01-14:2181,G01-15:2181,G01-16:2181"

csv_file_path = "file:///home/hduser/DrunkDetection/train_data48.csv"
predictor_path = "/home/hduser/DrunkDetection/shape_predictor_68_face_landmarks.dat"
model_path = "/home/hduser/DrunkDetection/rf48.pickle"
producer = KafkaProducer(bootstrap_servers='G01-01:9092', batch_size=163840,
                         buffer_memory=33554432, max_request_size=20485760)
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

# Spark session
spark = SparkSession \
    .builder \
    .appName("drunk streaming structure") \
    .getOrCreate()

log4j_logger = spark.sparkContext._jvm.org.apache.log4j
logger = log4j_logger.LogManager.getLogger(__name__)
logger.warn("logger start")


# process
@udf
def handler(s):
    s = s + "123123"
    return s


# Subscribe to 1 topic, read from kafka, length("value").alias("len"), "timestamp"
df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "G01-01:9092") \
    .option("subscribe", 'input') \
    .option("startingOffsets", "latest") \
    .load() \
    .select(decode("key", 'UTF-8').alias("key"), decode("value", "UTF-8").alias("value"))

rawQuery = df \
    .writeStream \
    .queryName("qraw") \
    .format("memory") \
    .trigger(processingTime='2 seconds') \
    .start()
rawQuery.show()
rawQuery.printSchema()

# Write key-value data from a DataFrame to a specific Kafka topic specified in an option
ds = df \
    .select(handler("key"), "value") \
    .writeStream \
    .format("console") \
    .trigger(continuous='1 second') \
    .start()

ds.awaitTermination()
