import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
age = np.random.normal(30, 10, 100).astype(int)
sex = np.random.randint(0, 2, 100)
BMI = np.random.normal(22, 3, 100)
child = np.random.randint(0, 3, 100)
smoke = np.random.randint(0, 2, 100)
region = np.random.randint(1, 5, 100)
charges = np.random.normal(10000, 500, 100).astype(int)

price = age * 150 + charges * 1 + child * 1500 + region * 500
price += np.where(sex == 0, 1000, 1500)
price += np.where(smoke == 0, 1500, 100)
price += np.where(BMI < 18.5, 4000, 0)
price += np.where((BMI >= 18.5) & (BMI < 25), 1000, 0)
price += np.where(BMI >= 25, 5000, 0)

df = pd.DataFrame({
    'age': age,
    'sex': sex,
    'bmi': BMI.astype(int),
    'children': child,
    'smoker': smoke,
    'region': region,
    'charges': charges,
    'price': price
})


X = df.drop('charges', axis=1)
y = df['charges']


model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

print(f"Intercept: {model.intercept_}")
for name, coef in zip(X.columns, model.coef_):
    print(f"{name}: {coef}")
print(f"R² Score: {r2_score(y, y_pred)}")
print(f"MSE: {mean_squared_error(y, y_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred))}")

plt.scatter(y, y_pred, alpha=0.6)
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs Predicted Charges')
plt.grid(True)
plt.show()
