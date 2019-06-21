from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split
from pyspark.sql.functions import decode

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
    .format("socket") \
    .option("host", "10.244.1.12") \
    .option("port", 23333) \
    .load()

# Write key-value data from a DataFrame to a specific Kafka topic specified in an option
ds = df \
    .selectExpr("key", "value") \
    .writeStream \
    .format("console") \
    .trigger(continuous='5 second') \
    .start()

ds.awaitTermination()
