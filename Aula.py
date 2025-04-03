import pandas as pd
import matplotlib.pyplot as plt

diabetes_df = pd.read_csv("diabetes_clean.csv")
print(diabetes_df.head())
X = diabetes_df.drop("glucose", axis=1).values
y = diabetes_df["glucose"].values
print(type(X), type(y))
X_bmi = X[:, 4]
print(y.shape, X_bmi.shape)
X_bmi = X_bmi.reshape(-1, 1)  # cria uma matriz 2D de uma única coluna.
print(y.shape, X_bmi.shape)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_bmi, y)
predictions = reg.predict(X_bmi)

plt.scatter(X_bmi, y)
plt.plot(X_bmi, predictions)
plt.ylabel("Blood Glucose (mg/dl)")
plt.xlabel("Body Mass Index")
plt.show()

diabetes_df[(diabetes_df["glucose"] > 0)
            & (diabetes_df[diabetes_df["glucose"]] > 0)  ]