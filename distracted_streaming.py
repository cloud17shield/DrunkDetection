from kafka import KafkaProducer

from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from keras.models import load_model
# import sparkdl as dl
import numpy as np
import cv2
import imutils
from keras.applications.vgg16 import preprocess_input
import tensorflow as tf

conf = SparkConf().setAppName("distract streaming").setMaster("yarn")
sc = SparkContext(conf=conf)
ssc = StreamingContext(sc, 0.5)
sql_sc = SQLContext(sc)
input_topic = 'input'
output_topic = 'output'
brokers = "G01-01:2181,G01-02:2181,G01-03:2181,G01-04:2181,G01-05:2181,G01-06:2181,G01-07:2181,G01-08:2181," \
          "G01-09:2181,G01-10:2181,G01-11:2181,G01-12:2181,G01-13:2181,G01-14:2181,G01-15:2181,G01-16:2181"

model_path = '/home/hduser/Distracted_vgg16_full.h5'  # /home/hduser/Distracted_vgg16_full.h5
model = load_model(model_path)
graph = tf.get_default_graph()
print(model.summary())


def my_decoder(s):
    return s


kafkaStream = KafkaUtils.createStream(ssc, brokers, 'test-consumer-group', {input_topic: 10},
                                      valueDecoder=my_decoder)
producer = KafkaProducer(bootstrap_servers='G01-01:9092', compression_type='gzip', batch_size=163840,
                         buffer_memory=33554432, max_request_size=20485760)


def handler(message):

    records = message.collect()
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
        image = cv2.imdecode(image, cv2.IMREAD_ANYCOLOR)
        print("img shape:", image.shape)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        image = image.reshape((-1, 224, 224, 3))
        print("img shape:", image.shape)
        image = preprocess_input(image)
        print("img shape:", image.shape)
        global graph
        global model
        with graph.as_default():
            ynew = model.predict_classes(image)
            print(type(ynew), ynew)


kafkaStream.foreachRDD(handler)
ssc.start()
ssc.awaitTermination()
