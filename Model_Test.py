import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import time

# Original dataset creation
# Simulating a dataset with predefined features for regression tasks.
data = {
    "Days_for_shipping_real": [3, 5, 4, 3, 2, 6, 2, 2, 3, 2, 6, 5, 4, 2, 2, 2, 5, 2, 0, 0,
                               5, 4, 3, 2, 6],
    "Days_for_shipment_scheduled": [4, 4, 4, 4, 4, 4, 1, 1, 2, 1, 2, 2, 2, 1, 1, 1, 2, 1,
                                    0, 0, 4, 2, 2, 2, 2],
    "Benefit_per_order": [91.25, -249.09, -247.78, 22.86, 134.21, 18.58, 95.18, 68.43, 133.72, 132.15,
                          130.58, 45.69, 21.76, 24.58, 16.39, -259.58, -246.36, 23.84, 102.26,
                          87.18, 154.86, 82.30, 22.37, 17.70, 90.28],
    "Sales_per_customer": [314.64, 311.36, 309.72, 304.81, 298.25, 294.98, 288.42, 285.14, 278.59,
                           275.31, 272.03, 268.76, 262.20, 245.81, 327.75, 324.47, 321.20,
                           317.92, 314.64, 311.36, 309.72, 304.81, 298.25, 294.98, 288.42]
}

df = pd.DataFrame(data)

# Augment the dataset with synthetic data
# Generating new samples by randomly assigning values to match the original features' ranges.
new_rows = []
for _ in range(1000):  # Adding 1000 new samples
    new_row = {
        "Days_for_shipping_real": np.random.randint(1, 7),  # Random days (1 to 6)
        "Days_for_shipment_scheduled": np.random.randint(0, 5),  # Random scheduled days (0 to 4)
        "Benefit_per_order": np.random.uniform(-300, 150)  # Random profit within a specified range
    }
    new_rows.append(new_row)

# Concatenating the new data into the original DataFrame
df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

# Adding a new feature: Total shipping days
# Combining two columns to create a derived feature.
df['Total_Days'] = df['Days_for_shipping_real'] + df['Days_for_shipment_scheduled']

# Checking for NaN values
if df.isnull().values.any():
    print("NaN values found. Handling missing data...")
    # Filling missing values with the mean of the respective column
    df.fillna(df.mean(), inplace=True)

# Splitting features and target variable
# Independent variables (X) and the dependent variable (y).
X = df[["Days_for_shipping_real", "Days_for_shipment_scheduled", "Benefit_per_order", "Total_Days"]]
y = df["Sales_per_customer"]

# Scaling the features
# Standardizing the feature set to have zero mean and unit variance for better model performance.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the data into training and testing sets
# 80% training data, 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initializing various regression models
models = {
    "Linear Regression": LinearRegression(),  # Simple linear regression
    "Decision Tree": DecisionTreeRegressor(),  # Decision Tree Regressor
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),  # Random Forest with 100 trees
    "Gradient Boosting": GradientBoostingRegressor()  # Gradient Boosting Regressor
}

# Training and evaluating each model
for model_name, model in models.items():
    start_time = time.time()  # Start timing the training process
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Predict on the test set
    end_time = time.time()  # End timing

    mse = mean_squared_error(y_test, y_pred)  # Calculate Mean Squared Error
    r2 = r2_score(y_test, y_pred)  # Calculate R² score
    duration = end_time - start_time  # Compute duration

    print(f"{model_name}:")
    print(f"  Mean Squared Error: {mse:.2f}")
    print(f"  R² Score: {r2:.2f}")
    print(f"  Training Duration: {duration:.4f} seconds\n")

# Hyperparameter tuning for Gradient Boosting Regressor
# Using GridSearchCV to find the optimal parameters for Gradient Boosting.
param_grid = {
    'n_estimators': [100, 200],  # Number of trees
    'max_depth': [3, 5, 7],  # Maximum depth of the tree
    'learning_rate': [0.01, 0.1, 0.2]  # Learning rate
}

grid_start_time = time.time()  # Start timing
grid_search = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=5)  # 5-fold cross-validation
grid_search.fit(X_train, y_train)  # Fit the grid search
grid_end_time = time.time()  # End timing

# Best model and its evaluation
best_model = grid_search.best_estimator_  # Retrieve the best model
y_pred_best = best_model.predict(X_test)  # Predict using the best model

best_mse = mean_squared_error(y_test, y_pred_best)  # Calculate MSE
best_r2 = r2_score(y_test, y_pred_best)  # Calculate R² score
grid_duration = grid_end_time - grid_start_time  # Compute duration for hyperparameter tuning

print("Best Gradient Boosting Regressor:")
print(f"  Mean Squared Error: {best_mse:.2f}")
print(f"  R² Score: {best_r2:.2f}")
print(f"  Hyperparameter Tuning Duration: {grid_duration:.4f} seconds\n")
