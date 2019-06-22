from pyspark.sql import SparkSession
from pyspark.sql.functions import *

input_topic = 'input'
output_topic = 'output'
brokers = "G01-01:2181,G01-02:2181,G01-03:2181,G01-04:2181,G01-05:2181,G01-06:2181,G01-07:2181,G01-08:2181," \
          "G01-09:2181,G01-10:2181,G01-11:2181,G01-12:2181,G01-13:2181,G01-14:2181,G01-15:2181,G01-16:2181 "

# Spark session
spark = SparkSession \
    .builder \
    .appName("drunk streaming structure") \
    .getOrCreate()

# Subscribe to 1 topic
df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "G01-01:9092") \
    .option("subscribe", 'input') \
    .load()


# rawQuery = df \
#     .writeStream \
#     .queryName("qraw") \
#     .format("memory") \
#     .trigger(continuous='1 second') \
#     .start()
#
# raw = spark.sql("select * from qraw")
# raw.show()

def handler(row):
    print(row)
    pass


# Write key-value data from a DataFrame to a specific Kafka topic specified in an option
ds = df \
    .select("timestamp", decode("key", 'UTF-8'), "value", length("value").alias("len")) \
    .writeStream \
    .foreach(handler)\
    .format("console") \
    .trigger(continuous='1 second') \
    .start()

ds.awaitTermination()
