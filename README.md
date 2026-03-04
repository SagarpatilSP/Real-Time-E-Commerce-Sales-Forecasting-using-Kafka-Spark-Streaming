Real-Time Sales Forecasting using Kafka and Spark
1. Project Summary

This project focuses on real-time data processing and sales prediction for an e-commerce system.
The data is streamed using Apache Kafka and processed in real time using Apache Spark Streaming.
After processing the data, a machine learning model predicts sales per customer.

2. Project Objective

The main goal of this project is to:

Process real-time e-commerce order data

Build a data streaming pipeline

Train a machine learning model for sales prediction

Generate analytics and visual insights

This helps businesses understand customer behavior and future sales trends.

3. Technologies Used

1. Apache Kafka
Kafka is used as a data streaming platform. It sends order and customer data continuously to the processing system.

2. Apache Spark
Spark consumes the streamed data from Kafka and performs real-time data processing and machine learning operations.

3. Python
Python is used for data preprocessing, model building, and visualization.

4. Dataset

The dataset used is the Smart Supply Chain Dataset from Kaggle.

Dataset size:

180,519 rows

53 columns

It contains information such as:

Order details

Shipping information

Product details

Customer information

Sales and profit data

This dataset helps analyze customer purchasing behavior and sales performance.

5. Project Pipeline

The workflow of the project is:

CSV Dataset
→ Data streamed using Kafka
→ Kafka sends data to Spark Streaming
→ Spark processes the data
→ Machine Learning model predicts sales
→ Results are visualized using Matplotlib

6. Data Analysis & Visualization

We used Matplotlib to visualize the model performance.

A scatter plot compares:

Actual Sales per Customer

Predicted Sales

A reference line f(x) = x represents perfect prediction.

If prediction points are close to this line, the model prediction is accurate.

7. Model Training
Feature Selection

The following features were used to train the model:

Days_for_shipping_real

Days_for_shipment_scheduled

Benefit_per_order

Order_Item_Quantity

Order_Item_Discount

Product_Price

These features were combined into a feature vector.

Model Used

We used Linear Regression to predict:

Sales_per_customer

This model was selected after testing multiple models and choosing the one that gave good accuracy with faster training time.

8. Prediction Process

For each incoming data batch:

Data is split into 80% training and 20% testing

The model is trained using the training dataset

Predictions are generated on the test dataset

Actual and predicted values are collected for evaluation

9. Results

The model predicts Sales_per_customer for each order.

To evaluate accuracy:

A perfect prediction line (f(x)=x) is plotted

Error tolerance of ±15% is used

Two boundary lines were added:

f(x) = 0.85x

f(x) = 1.15x

Predictions falling between these lines are considered acceptable predictions"# Real-Time-E-Commerce-Sales-Forecasting-using-Kafka-Spark-Streaming" 
