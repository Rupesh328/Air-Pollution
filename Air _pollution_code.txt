import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('air_quality.csv')  # Replace with your filename

# View data
print(df.head())

# Drop rows with missing values
df = df.dropna()

# Feature selection
features = ['PM2.5', 'PM10', 'NO2', 'CO', 'Temperature', 'Humidity', 'WindSpeed', 'Traffic']
target = 'AQI'

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(y_test.values[:100], label='Actual AQI')
plt.plot(y_pred[:100], label='Predicted AQI')
plt.title('Actual vs Predicted AQI')
plt.legend()
plt.show()
