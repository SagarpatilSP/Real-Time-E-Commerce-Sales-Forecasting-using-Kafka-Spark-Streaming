import os
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET = os.getenv("BUCKET_NAME")

spark = SparkSession.builder \
    .appName("Customer Sales ML Model") \
    .config("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY) \
    .config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_KEY) \
    .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
    .getOrCreate()

df = spark.read.parquet(f"s3a://{BUCKET}/gold/customer_sales")

feature_cols = ["avg_customer_sales"]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

ml_df = assembler.transform(df)

train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="total_customer_sales",
    numTrees=100
)

model = rf.fit(train_df)

predictions = model.transform(test_df)

evaluator = RegressionEvaluator(
    labelCol="total_customer_sales",
    predictionCol="prediction",
    metricName="rmse"
)

print("RMSE:", evaluator.evaluate(predictions))

model.write().overwrite().save(
    f"s3a://{BUCKET}/models/customer_sales_model"
)

spark.stop()
