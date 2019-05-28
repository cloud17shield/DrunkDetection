from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import pickle
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import numpy as np
import imutils
import dlib
import cv2
#import pydoop.hdfs as hdfs

conf = SparkConf().setAppName("drunk test").setMaster("yarn")
sc = SparkContext(conf=conf)
#sc.setLogLevel("INFO")
print("drunk prediction test")
sqlCtx = SQLContext(sc)
print("drunk prediction test")
