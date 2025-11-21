import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def fill_null_with_rand_values(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Fills null values in the datafarme with random values from the respective columns

    ## Parameters:
    df: The dataframe to be cleaned

    ## Returns:
    The dataframe with null values imputed
    '''
    for col in df.columns.to_list():
        size = df[col].isna().sum()
        if size == 0:
            print("No null values in", col)
            continue
        non_null_values = df[col].dropna()
        rand_vals = np.random.choice(
            non_null_values, size=size, replace=True)
        df.loc[df[col].isna(), col] = rand_vals
    return df


df = pd.read_csv("ecommerce_customer_data.csv")
print(df.info())

df = df.drop(columns=['FavoriteCategory', 'SecondFavoriteCategory'])

# print(df.isna().sum())

# Cleaning CustomerID by trimming the values and dropping rows with nulls
df["CustomerID"] = df["CustomerID"].str[4:]
df = df.dropna(subset=["CustomerID"])
df["CustomerID"] = df["CustomerID"].astype(int)

# Imputing null values with random values from the respective columns
df = fill_null_with_rand_values(df)

# Cleaning Registration Date by change dtpes and filling nulls
df["RegistrationDate"] = pd.to_datetime(df["RegistrationDate"])

# Imputing improper values in Age with random values of its own distribution
df["Age"] = df["Age"].astype(int)

fig, axes = plt.subplots(1, 2)
axes[0].hist(df["Age"])

data = df["Age"][df["Age"] >= 18]
size = df["Age"][df["Age"] < 18].count()
df.loc[df["Age"] < 18, "Age"] = np.random.choice(data, size=size, replace=True)

axes[1].hist(df["Age"])
plt.show()

for col in df.select_dtypes(include='object'):
    print(col, ":")
    print(df[col].unique())
# Standardizing categories in columns
df.loc[df["Gender"] == 'Male', "Gender"] = 'M'
df.loc[df["Gender"] == 'Female', "Gender"] = 'F'

df.loc[df["IncomeLevel"] == 'L', "IncomeLevel"] = 'Low'
df.loc[df["IncomeLevel"] == 'H', "IncomeLevel"] = 'High'

num_cols = df.select_dtypes(include=['int', 'float']).columns
num_rows = int(np.ceil(len(num_cols)/4))
fig, axes = plt.subplots(num_rows, 4, layout='constrained',
                         figsize=(len(num_cols)*4, num_rows*3))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    axes[i].hist(df[col])
    axes[i].set_title(col)

plt.show()
