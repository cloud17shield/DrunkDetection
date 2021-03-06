from kafka import KafkaProducer

from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

import sparkdl as dl
import numpy as np
import cv2
import imutils
from keras.applications.vgg16 import preprocess_input

conf = SparkConf().setAppName("distract streaming").setMaster("yarn")
sc = SparkContext(conf=conf)
ssc = StreamingContext(sc, 0.5)
sql_sc = SQLContext(sc)
input_topic = 'input'
output_topic = 'output'
brokers = "G01-01:2181,G01-02:2181,G01-03:2181,G01-04:2181,G01-05:2181,G01-06:2181,G01-07:2181,G01-08:2181," \
          "G01-09:2181,G01-10:2181,G01-11:2181,G01-12:2181,G01-13:2181,G01-14:2181,G01-15:2181,G01-16:2181"

model_path = '/home/hduser/Distracted_vgg16_full.h5'  # /home/hduser/Distracted_vgg16_full.h5


def my_decoder(s):
    return bytearray(s)


def loadAndPreprocessKeras(uri):
    image = np.asarray(uri, dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_ANYCOLOR)
    print("image type:", image.shape)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    print("image type:", image.shape)
    image = image.reshape((-1, 224, 224, 3))
    print("image type:", image.shape)
    return preprocess_input(image)


kafkaStream = KafkaUtils.createStream(ssc, brokers, 'test-consumer-group', {input_topic: 10},
                                      valueDecoder=my_decoder)
producer = KafkaProducer(bootstrap_servers='G01-01:9092', compression_type='gzip', batch_size=163840,
                         buffer_memory=33554432, max_request_size=20485760)

transformer = dl.KerasImageFileTransformer(inputCol="_2", outputCol="predictions",
                                           modelFile=model_path,  # local file path for model
                                           imageLoader=loadAndPreprocessKeras,
                                           outputMode="vector")


def handler(message):
    try:
        records = message.toDF()
        print("Schema()")
        # records.printSchema()
        # records.show()
        keras_pred_df = transformer.transform(records)
        keras_pred_df.printSchema()
        keras_pred_df.show()
    except Exception as e:
        print("ErrorType:", e)
    # for record in records:
    #     try:
    #         print('record', len(record), type(record))
    #         print('-----------')
    #         print('tuple', type(recorbytearray(value)d[0]), type(record[1]))
    #     except Exception:
    #         print("error")
    #     # producer.send(output_topic, b'message received')
    #     key = record[0]
    #     value = record[1]
    #     print("len", len(key), len(value))
    #
    #     print("start processing")
    #     image = np.asarray(, dtype="uint8")
    #     # image = np.frombuffer(value, dtype=np.uint8)
    #     # img = image.reshape(300, 400, 3)
    #     # img = cv2.imread("/tmp/" + key)
    #     img = cv2.imdecode(image, cv2.IMREAD_ANYCOLOR)
    #     frame = imutils.resize(img, width=450)
    #     print('img shape', frame.shape)


kafkaStream.foreachRDD(handler)
ssc.start()
ssc.awaitTermination()
