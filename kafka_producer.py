from kafka import KafkaProducer
from flask import Flask
import pandas as pd
import json
import threading

# Kafka Producer settings
# Creating a Producer to send data to Kafka.
# bootstrap_servers: Specifies the address of the Kafka Broker.
# value_serializer: Converts the data into JSON format and encodes it.
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',  # Kafka Broker address (running on localhost)
    value_serializer=lambda v: json.dumps(v).encode('utf-8')  # Serialize data to JSON format
)

# Initializing the Flask application.
app = Flask(__name__)

# Function to read data from a CSV file and send it to Kafka
def produce_csv_data():
    # Reading a local CSV file. Example file name: 'data.csv'.
    # The encoding parameter is used to prevent character encoding issues (ISO-8859-1).
    data = pd.read_csv('C:/Users/90533/Desktop/data.csv', encoding='ISO-8859-1')

    # Iterating through the data to send each row to Kafka.
    for _, row in data.iterrows():
        # Converting the row into a dictionary format.
        order_data = {
            'Type': row['Type'],  # Order type information
            'Days_for_shipping_real': row['Days for shipping (real)'],  # Actual shipping days
            'Days_for_shipment_scheduled': row['Days for shipment (scheduled)'],  # Scheduled shipping days
            'Benefit_per_order': row['Benefit per order'],  # Profit per order
            'Sales_per_customer': row['Sales per customer'],  # Sales per customer
            'Delivery_Status': row['Delivery Status'],  # Delivery status
            'Late_delivery_risk': row['Late_delivery_risk'],  # Risk of late delivery
            'Category_Id': row['Category Id'],  # Category ID
            'Category_Name': row['Category Name'],  # Category name
            'Customer_City': row['Customer City'],  # Customer city
            'Customer_Country': row['Customer Country'],  # Customer country
            'Customer_Email': row['Customer Email'],  # Customer email address
            'Customer_Fname': row['Customer Fname'],  # Customer first name
            'Customer_Id': row['Customer Id'],  # Customer ID
            'Customer_Lname': row['Customer Lname'],  # Customer last name
            'Customer_Password': row['Customer Password'],  # Customer password
            'Customer_Segment': row['Customer Segment'],  # Customer segment
            'Customer_State': row['Customer State'],  # Customer state
            'Customer_Street': row['Customer Street'],  # Customer street
            'Customer_Zipcode': row['Customer Zipcode'],  # Customer zip code
            'Department_Id': row['Department Id'],  # Department ID
            'Department_Name': row['Department Name'],  # Department name
            'Latitude': row['Latitude'],  # Latitude information
            'Longitude': row['Longitude'],  # Longitude information
            'Market': row['Market'],  # Market information
            'Order_City': row['Order City'],  # Order city
            'Order_Country': row['Order Country'],  # Order country
            'Order_Customer_Id': row['Order Customer Id'],  # Customer ID who placed the order
            'Order_Date': row['order date (DateOrders)'],  # Order date
            'Order_Id': row['Order Id'],  # Order ID
            'Order_Item_Cardprod_Id': row['Order Item Cardprod Id'],  # Product card ID
            'Order_Item_Discount': row['Order Item Discount'],  # Product discount
            'Order_Item_Discount_Rate': row['Order Item Discount Rate'],  # Discount rate percentage
            'Order_Item_Id': row['Order Item Id'],  # Order item ID
            'Order_Item_Product_Price': row['Order Item Product Price'],  # Product price
            'Order_Item_Profit_Ratio': row['Order Item Profit Ratio'],  # Profit ratio
            'Order_Item_Quantity': row['Order Item Quantity'],  # Product quantity
            'Sales': row['Sales'],  # Sales amount
            'Order_Item_Total': row['Order Item Total'],  # Total order amount
            'Order_Profit_Per_Order': row['Order Profit Per Order'],  # Profit per order
            'Order_Region': row['Order Region'],  # Order region
            'Order_State': row['Order State'],  # Order state
            'Order_Status': row['Order Status'],  # Order status
            'Order_Zipcode': row['Order Zipcode'],  # Order zip code
            'Product_Card_Id': row['Product Card Id'],  # Product card ID
            'Product_Category_Id': row['Product Category Id'],  # Product category ID
            'Product_Description': row['Product Description'],  # Product description
            'Product_Image': row['Product Image'],  # Product image
            'Product_Name': row['Product Name'],  # Product name
            'Product_Price': row['Product Price'],  # Product price
            'Product_Status': row['Product Status'],  # Product status
            'Shipping_Date': row['shipping date (DateOrders)'],  # Shipping date
            'Shipping_Mode': row['Shipping Mode']  # Shipping mode
        }

        # Sending data to Kafka
        try:
            producer.send('deneme2', order_data)  # Sending data to Kafka topic 'deneme2'.
            print(f"Data sent: {order_data}")  # Logging the sent data to the console.
        except Exception as e:
            print(f"Message sending error: {e}")  # Logging any errors during sending.
        # time.sleep(0.01)  # Optional delay between sends can be added.

if __name__ == '__main__':
    # Running the Kafka producer in a separate thread.
    threading.Thread(target=produce_csv_data, daemon=True).start()
    # Starting the Flask application.
    app.run(debug=True, port=5000)  # Flask application running on port 5000.
