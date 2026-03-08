from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, dayofmonth

spark = SparkSession.builder \
    .appName("Silver Layer Processing") \
    .config("spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3") \
    .getOrCreate()

# Read bronze data as stream
bronze_df = spark.readStream \
    .format("parquet") \
    .load("s3a://my-data-lake/bronze/orders")

# Data cleaning
clean_df = bronze_df \
    .filter(col("Customer_Id").isNotNull()) \
    .filter(col("Sales") > 0) \
    .filter(col("Product_Price") > 0) \
    .filter(col("Order_Item_Quantity") > 0) \
    .dropDuplicates(["event_id"])

# Add partition columns
clean_df = clean_df \
    .withColumn("year", year(col("Order_Date"))) \
    .withColumn("month", month(col("Order_Date"))) \
    .withColumn("day", dayofmonth(col("Order_Date")))

# Write to Silver layer
query = clean_df.writeStream \
    .format("parquet") \
    .outputMode("append") \
    .option("path", "s3a://my-data-lake/silver/orders") \
    .option("checkpointLocation", "s3a://my-data-lake/checkpoints/silver_orders") \
    .partitionBy("year", "month", "day") \
    .start()

query.awaitTermination()

