from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, current_timestamp
from dotenv import load_dotenv
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET = os.getenv("BUCKET_NAME")


spark = SparkSession.builder \
    .appName("Gold Layer Processing") \
    .config("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY)\
    .config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_KEY)\
    .getOrCreate()

# ------------------------------
# Read Silver Layer (Streaming)
# ------------------------------

silver_df = spark.readStream.parquet(
    "s3a://my-data-lake/silver/orders"
)

# ------------------------------
# 1 Sales Summary
# ------------------------------

sales_summary_df = silver_df.groupBy("Order_Date").agg(
    sum("Sales").alias("total_sales"),
    avg("Product_Price").alias("avg_product_price")
).withColumn("processed_time", current_timestamp())

sales_query = sales_summary_df.writeStream \
    .format("parquet") \
    .outputMode("complete") \
    .option("path", "s3a://my-data-lake/gold/sales_summary") \
    .option("checkpointLocation", "s3a://my-data-lake/checkpoints/gold_sales") \
    .start()

# ------------------------------
# 2 Customer Sales Summary
# ------------------------------

customer_sales_df = silver_df.groupBy("Customer_Id").agg(
    sum("Sales").alias("total_customer_sales"),
    avg("Sales").alias("avg_customer_sales")
).withColumn("processed_time", current_timestamp())

customer_query = customer_sales_df.writeStream \
    .format("parquet") \
    .outputMode("complete") \
    .option("path", "s3a://my-data-lake/gold/customer_sales") \
    .option("checkpointLocation", "s3a://my-data-lake/checkpoints/gold_customer") \
    .start()

# ------------------------------
# 3 Shipping Performance
# ------------------------------

shipping_df = silver_df.groupBy("Shipping_Mode").agg(
    avg("Days_for_shipping_real").alias("avg_shipping_days"),
    avg("Days_for_shipment_scheduled").alias("avg_scheduled_days")
).withColumn("processed_time", current_timestamp())

shipping_query = shipping_df.writeStream \
    .format("parquet") \
    .outputMode("complete") \
    .option("path", "s3a://my-data-lake/gold/shipping_performance") \
    .option("checkpointLocation", "s3a://my-data-lake/checkpoints/gold_shipping") \
    .start()

spark.streams.awaitAnyTermination()
