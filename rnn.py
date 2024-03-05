import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_excel('Book3.xlsx')

# Preprocessing
# Assuming you need to preprocess the data, such as handling missing values, scaling, etc.
# For simplicity, we'll just drop missing values and use the required features and target variable
data.dropna(inplace=True)
X = data[['Voltage', 'Temp', 'Current']]  # Features
y = data['RUL']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

print(f'Training Mean Squared Error: {mse_train}')
print(f'Testing Mean Squared Error: {mse_test}')

# Now, let's predict RUL for new batteries
# Assuming you have a new dataset containing features for new batteries
new_battery_data = pd.read_excel('Book2.xlsx')

# Preprocess new data (similar to preprocessing steps above)
# For simplicity, we'll assume the new data is already preprocessed and contains the same features as the training data

# Predict RUL for new batteries
new_battery_features = new_battery_data[['Voltage', 'Temp', 'Current']]
new_battery_predictions = model.predict(new_battery_features)

# Print predicted RUL for new batteries
print('Predicted RUL for new batteries:')
for i, pred in enumerate(new_battery_predictions):
    print(f'Battery {i+1}: {pred}')