# Import libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -----------------------------
# Task 1: Create Dataset
# -----------------------------
np.random.seed(42)

n = 60  # at least 50 records

area_sqft = np.random.randint(500, 3000, n)
num_bedrooms = np.random.randint(1, 5, n)
age_years = np.random.randint(0, 30, n)

# Create target (price in lakhs)
price_lakhs = (
    area_sqft * 0.05 +
    num_bedrooms * 5 -
    age_years * 0.3 +
    np.random.normal(0, 5, n)  # noise
)

# Create DataFrame
df = pd.DataFrame({
    "area_sqft": area_sqft,
    "num_bedrooms": num_bedrooms,
    "age_years": age_years,
    "price_lakhs": price_lakhs
})

# Features & target
X = df[["area_sqft", "num_bedrooms", "age_years"]]
y = df["price_lakhs"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Print intercept & coefficients
print("Intercept:", model.intercept_)
print("Coefficients:")
for col, coef in zip(X.columns, model.coef_):
    print(f"{col}: {coef}")

# Predictions
y_pred = model.predict(X)

# Show first 5 actual vs predicted
result = pd.DataFrame({
    "Actual": y.head(),
    "Predicted": y_pred[:5]
})
print("\nFirst 5 Actual vs Predicted:\n", result)

# -----------------------------
# Task 2: Evaluation
# -----------------------------
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print("\nMAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

# Comment (IMPORTANT for submission)
# MAE shows average absolute error in predictions.
# RMSE penalizes larger errors more heavily than MAE.
# R2 shows how much variance in the target is explained by the model.

# -----------------------------
# Task 3: Residuals
# -----------------------------
residuals = y - y_pred

# Plot histogram
plt.figure()
plt.hist(residuals)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# Comment:
# Residual = Actual - Predicted value.
# A symmetric histogram around zero suggests the model is performing well without bias.