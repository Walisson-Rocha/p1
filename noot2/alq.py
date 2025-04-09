import pandas as pd

diabetes = pd.read_csv('./diabetes_clean.csv')

diabetes_df = diabetes[(diabetes['glucose'] > 0) & 
                        (diabetes['insulin'] > 0) & 
                        (diabetes['bmi'] > 30)]


print(diabetes_df)

x = diabetes_df.drop("glucose", axis=1).values
y = diabetes_df["glucose"].values
print(x.shape, y.shape)