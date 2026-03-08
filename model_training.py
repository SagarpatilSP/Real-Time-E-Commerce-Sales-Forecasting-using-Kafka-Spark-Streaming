from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from dotenv import load_dotenv
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET = os.getenv("BUCKET_NAME")

spark = SparkSession.builder \
    .appName("Customer Sales ML Model") \
    .config("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY)\
    .config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_KEY)\
    .getOrCreate()


# Load Gold Dataset
df = spark.read.parquet(
    "s3a://my-data-lake/gold/customer_sales"
)

# Feature Columns
feature_cols = [
    "total_customer_sales",
    "avg_customer_sales"
]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

ml_df = assembler.transform(df)

# Train Test Split
train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)

# Model
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="total_customer_sales",
    numTrees=100,
    maxDepth=10
)

model = rf.fit(train_df)

# Prediction
predictions = model.transform(test_df)

predictions.select(
    "Customer_Id",
    "total_customer_sales",
    "prediction"
).show()


# Evaluation
evaluator = RegressionEvaluator(
    labelCol="total_customer_sales",
    predictionCol="prediction",
    metricName="rmse"
)

rmse = evaluator.evaluate(predictions)

print("Model RMSE:", rmse)

# Save Model
model.write().overwrite().save(
    "s3a://my-data-lake/models/customer_sales_model"
)

spark.stop()
