import shutil
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, DateType
from pyspark.sql import SparkSession
from pyspark.sql.types import TimestampType
from pyspark.sql.functions import from_json, col, current_timestamp, year, month, dayofmonth
import os
from dotenv import load_dotenv
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET = os.getenv("BUCKET_NAME")

# Function to start the Spark streaming session
def start_spark_streaming():
    # Initialize a Spark session with Kafka package for streaming
    spark = SparkSession.builder \
        .appName("Bronze") \
        .config("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY)\
        .config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_KEY)\
        .getOrCreate()

    # Define the schema of the incoming data
    sensor_schema = StructType([
        StructField("Type", StringType(), True),
        StructField("Days_for_shipping_real", IntegerType(), True),
        StructField("Days_for_shipment_scheduled", IntegerType(), True),
        StructField("Benefit_per_order", FloatType(), True),
        StructField("Sales_per_customer", FloatType(), True),
        StructField("Delivery_Status", StringType(), True),
        StructField("Late_delivery_risk", IntegerType(), True),
        StructField("Category_Id", IntegerType(), True),
        StructField("Category_Name", StringType(), True),
        StructField("Customer_City", StringType(), True),
        StructField("Customer_Country", StringType(), True),
        StructField("Customer_Email", StringType(), True),
        StructField("Customer_Fname", StringType(), True),
        StructField("Customer_Id", IntegerType(), True),
        StructField("Customer_Lname", StringType(), True),
        StructField("Customer_Password", StringType(), True),
        StructField("Customer_Segment", StringType(), True),
        StructField("Customer_State", StringType(), True),
        StructField("Customer_Street", StringType(), True),
        StructField("Customer_Zipcode", StringType(), True),
        StructField("Department_Id", IntegerType(), True),
        StructField("Department_Name", StringType(), True),
        StructField("Latitude", FloatType(), True),
        StructField("Longitude", FloatType(), True),
        StructField("Market", StringType(), True),
        StructField("Order_City", StringType(), True),
        StructField("Order_Country", StringType(), True),
        StructField("Order_Customer_Id", IntegerType(), True),
        StructField("Order_Date", DateType(), True),
        StructField("Order_Id", IntegerType(), True),
        StructField("Order_Item_Cardprod_Id", IntegerType(), True),
        StructField("Order_Item_Discount", FloatType(), True),
        StructField("Order_Item_Discount_Rate", FloatType(), True),
        StructField("Order_Item_Id", IntegerType(), True),
        StructField("Order_Item_Product_Price", FloatType(), True),
        StructField("Order_Item_Profit_Ratio", FloatType(), True),
        StructField("Order_Item_Quantity", IntegerType(), True),
        StructField("Sales", FloatType(), True),
        StructField("Order_Item_Total", FloatType(), True),
        StructField("Order_Profit_Per_Order", FloatType(), True),
        StructField("Order_Region", StringType(), True),
        StructField("Order_State", StringType(), True),
        StructField("Order_Status", StringType(), True),
        StructField("Order_Zipcode", StringType(), True),
        StructField("Product_Card_Id", IntegerType(), True),
        StructField("Product_Category_Id", IntegerType(), True),
        StructField("Product_Description", StringType(), True),
        StructField("Product_Image", StringType(), True),
        StructField("Product_Name", StringType(), True),
        StructField("Product_Price", FloatType(), True),
        StructField("Product_Status", StringType(), True),
        StructField("Shipping_Date", DateType(), True),
        StructField("Shipping_Mode", StringType(), True),

        StructField("Event_id", StringType(), True),
        StructField("Event_time", TimestampType(), True)
    ])


    # Read data from Kafka    
    kafka_df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "Mytopic") \
        .load()

    # Convert key and value columns to strings
    value_df = kafka_df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")

    # Parse JSON data
    json_df = value_df.select(from_json(col("value").cast("string"), sensor_schema).alias("data")).select("data.*")

    bronze_df = json_df.withColumn("ingestion_time", current_timestamp())

    bronze_df = bronze_df \
        .withColumn("year", year(col("Event_time"))) \
        .withColumn("month", month(col("Event_time"))) \
        .withColumn("day", dayofmonth(col("Event_time")))

    query = bronze_df \
        .writeStream \
        .format("parquet") \
        .outputMode("append") \
        .option("path", "s3a://my-data-lake/bronze/orders") \
        .option("checkpointLocation", "s3a://my-data-lake/checkpoints/bronze_orders") \
        .partitionBy("year", "month", "day") \
        .start()

    query.awaitTermination()

# Run the streaming function when the script is executed directly
if __name__ == '__main__':
    start_spark_streaming()
