import os
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, month, dayofweek
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET = os.getenv("BUCKET_NAME")

spark = SparkSession.builder \
    .appName("Sales Forecasting Model") \
    .config("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY) \
    .config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_KEY) \
    .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
    .getOrCreate()


# -----------------------------
# 1 Read Gold Layer
# -----------------------------

df = spark.read.parquet(f"s3a://{BUCKET}/gold/orders")


# -----------------------------
# 2 Feature Engineering
# -----------------------------

df = df.withColumn("order_month", month(col("Order_Date"))) \
       .withColumn("order_dayofweek", dayofweek(col("Order_Date")))


# -----------------------------
# 3 Encode Categorical Columns
# -----------------------------

segment_indexer = StringIndexer(
    inputCol="Customer_Segment",
    outputCol="segment_index"
)

market_indexer = StringIndexer(
    inputCol="Market",
    outputCol="market_index"
)

shipping_indexer = StringIndexer(
    inputCol="Shipping_Mode",
    outputCol="shipping_index"
)

df = segment_indexer.fit(df).transform(df)
df = market_indexer.fit(df).transform(df)
df = shipping_indexer.fit(df).transform(df)


# -----------------------------
# 4 Select Features
# -----------------------------

feature_cols = [
    "Product_Price",
    "Order_Item_Quantity",
    "Order_Item_Discount",
    "Order_Item_Discount_Rate",
    "segment_index",
    "market_index",
    "shipping_index",
    "order_month",
    "order_dayofweek"
]


assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

ml_df = assembler.transform(df)


# -----------------------------
# 5 Train Test Split
# -----------------------------

train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)


# -----------------------------
# 6 Random Forest Model
# -----------------------------

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="Sales",
    numTrees=100,
    maxDepth=10
)

model = rf.fit(train_df)


# -----------------------------
# 7 Predictions
# -----------------------------

predictions = model.transform(test_df)


# -----------------------------
# 8 Model Evaluation
# -----------------------------

evaluator = RegressionEvaluator(
    labelCol="Sales",
    predictionCol="prediction",
    metricName="rmse"
)

rmse = evaluator.evaluate(predictions)

print("RMSE:", rmse)


# -----------------------------
# 9 Save Model to S3
# -----------------------------

model.write().overwrite().save(
    f"s3a://{BUCKET}/models/sales_forecasting_model"
)


spark.stop()
