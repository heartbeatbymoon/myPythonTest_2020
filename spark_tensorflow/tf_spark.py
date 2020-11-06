# encoding=utf-8
import pyspark
import tensorflow as tf
import pandas as pd
import numpy as np
from os.path import abspath
from pyspark.sql import SparkSession
from sklearn.linear_model import LinearRegression

# warehouse_location points to the default location for managed databases and tables
warehouse_location = abspath('spark-warehouse')

spark = SparkSession.builder \
    .appName("Python Spark SQL Hive integration example") \
    .master("local[2]") \
    .config("hive.metastore.uris", "thrift://47.115.55.203:9083") \
    .config("dfs.client.use.datanode.hostname", "true") \
    .config("mapreduce.input.fileinputformat.input.dir.recursive", "true") \
    .config("hive.input.dir.recursive", "true") \
    .config("hive.mapred.supports.subdirectories", "true") \
    .config("hive.supports.subdirectories", "true") \
    .config("spark.driver.maxResultSize", "5g") \
    .enableHiveSupport() \
    .getOrCreate()

data = spark.sql("select * from default.loggen").toPandas()
print(data)
print(type(data))

# 注意，不要 .show()后  再toPandas() ,否则报错
feature = data['up'].values.reshape(-1, 1)
label = data['down'].values.reshape(-1, 1)

regression = LinearRegression()
model = regression.fit(feature, label)  # 可以看出 传入的都是  二维列向量
# 斜率 m
print(model.coef_)

# 截距 b
print(model.intercept_)

spark.stop()
