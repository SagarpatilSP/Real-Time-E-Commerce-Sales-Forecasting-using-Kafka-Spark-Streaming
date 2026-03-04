import shutil
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, DateType
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# List to store predictions for analysis and visualization
predictions_list = []

# Folder to store plot outputs
output_folder = "plots"

# If the output folder exists, delete and recreate it
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

# Global variables for batch counter and model
batch_counter = 0
lr_model = None

# Function to start the Spark streaming session
def start_spark_streaming():
    global batch_counter, lr_model  # Define global variables here

    # Initialize a Spark session with Kafka package for streaming
    spark = SparkSession.builder \
        .appName("KafkaSparkStreaming") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3") \
        .getOrCreate()

    # Print Kafka server and topic information
    print("Kafka bootstrap servers:", spark.conf.get("kafka.bootstrap.servers", "localhost:9092"))
    print("Kafka topic:", "deneme2")

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
        StructField("Shipping_Mode", StringType(), True)
    ])

    # Read data from Kafka
    df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "deneme2") \
        .load()

    # Convert key and value columns to strings
    df = df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")

    # Parse JSON data
    json_df = df.select(from_json(col("value").cast("string"), sensor_schema).alias("data")).select("data.*")

    # Combine features for model input using VectorAssembler
    assembler = VectorAssembler(
        inputCols=["Days_for_shipping_real", "Days_for_shipment_scheduled",
                   "Benefit_per_order", "Order_Item_Quantity", "Order_Item_Discount",
                   "Product_Price"],
        outputCol="features"
    )

    # Define a Linear Regression model
    lr = LinearRegression(featuresCol="features", labelCol="Sales_per_customer")

    # Function to process each data batch
    def model_prediction(batch_df, batch_id):
        global batch_counter, lr_model  # Specify these variables as global
        if not batch_df.isEmpty():
            batch_counter += 1

            # Split data into training and testing sets
            train_df, test_df = batch_df.randomSplit([0.8, 0.2])
            train_df = assembler.transform(train_df)

            # Retrain the model every 10 batches
            if batch_counter % 10 == 0:
                print(f"Retraining the model at batch {batch_counter}")
                lr_model = lr.fit(train_df)

            # If model exists, make predictions on the test data
            if lr_model is not None:
                test_df = assembler.transform(test_df)
                predictions = lr_model.transform(test_df)
                predictions_list.extend(predictions.select("prediction", "Sales_per_customer").collect())

                # Visualization and error calculations
                if len(predictions_list) > 0:
                    predictions_df = pd.DataFrame(predictions_list, columns=["prediction", "Sales_per_customer"])
                    predictions_df['actual'] = predictions_df['Sales_per_customer']
                    predictions_df['difference'] = predictions_df['actual'] - predictions_df['prediction']
                    predictions_df['percentage_error'] = (predictions_df['difference'] / predictions_df['actual']) * 100

                    print(predictions_df[['Sales_per_customer', 'prediction', 'difference', 'percentage_error']])

                    plt.clf()  # Clear the plot

                    # Points with error below 15%
                    low_error_mask = (abs(predictions_df['difference']) / predictions_df['actual']) < 0.15
                    num_low_error = low_error_mask.sum()
                    num_other = len(predictions_df) - num_low_error

                    # Scatter plot for points with low and high error
                    plt.scatter(predictions_df.loc[low_error_mask, 'Sales_per_customer'],
                                predictions_df.loc[low_error_mask, 'prediction'],
                                color='orange', label='Error < 15%', alpha=0.5)

                    plt.scatter(predictions_df.loc[~low_error_mask, 'Sales_per_customer'],
                                predictions_df.loc[~low_error_mask, 'prediction'],
                                color='blue', alpha=0.5, label='Other Points')

                    plt.title('Sales per Customer vs Prediction')
                    plt.xlabel('Sales per Customer')
                    plt.ylabel('Prediction')

                    # Line of best fit
                    x_values = np.linspace(min(predictions_df['Sales_per_customer']),
                                           max(predictions_df['Sales_per_customer']), 100)
                    plt.plot(x_values, x_values, color='red', linestyle='--', label='F(x) = x')

                    # Lines representing 15% error boundaries
                    plt.plot(x_values, x_values * 1.15, color='green', linestyle=':',
                             label='F(x) = 1.15x (15% above)')
                    plt.plot(x_values, x_values * 0.85, color='blue', linestyle=':', label='F(x) = 0.85x (15% below)')

                    # Display point counts for low and high error
                    plt.text(0.05, 0.95, f'Error < 15%: {num_low_error}', transform=plt.gca().transAxes,
                             fontsize=12, verticalalignment='top', color='orange')
                    plt.text(0.05, 0.90, f'Other Points: {num_other}', transform=plt.gca().transAxes,
                             fontsize=12, verticalalignment='top', color='blue')

                    plt.legend()
                    plt.savefig(f'{output_folder}/plot_batch_{batch_id}.png')
                    plt.close()  # Close the plot

    try:
        # Start the streaming query and apply batch processing function
        print("Starting the stream...")
        plt.ion()
        query = json_df.writeStream \
            .foreachBatch(model_prediction) \
            .outputMode("append") \
            .start()

        query.awaitTermination()  # Wait for termination

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        spark.stop()  # Stop the Spark session upon completion

# Run the streaming function when the script is executed directly
if __name__ == '__main__':
    start_spark_streaming()