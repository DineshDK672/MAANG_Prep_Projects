import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("Data Wrangler Pipeline")
st.write("This dashboard shows the statistics of a ecommerce customer database before cleaning and after cleaning using the data wrangler pipeline")


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


def display_stats(df: pd.DataFrame) -> None:

    # Displaying null values
    nulls = df.isnull().sum()
    plt.bar(nulls.index.to_list(), nulls.values)
    plt.title("Distribution of Null values across Columns")
    plt.tick_params(axis='x', labelrotation=90)
    plt.xlabel("Columns")
    plt.ylabel("Null Values")
    plt.tight_layout()
    st.pyplot(plt)

    ignore_cols = ["CustomerID", "RegistrationDate",
                   'FavoriteCategory', 'SecondFavoriteCategory']
    for col in ignore_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Displaying Column Distribution in Streamlit
    # Categorical columns
    cat_cols = df.select_dtypes(include='object').columns
    cat_cols = cat_cols[1:]
    cat_rows = int(np.ceil(len(cat_cols)/2))
    fig, ax = plt.subplots(4, 2, figsize=(cat_rows*3, 8), layout="constrained")
    ax = ax.flatten()

    for i, col in enumerate(cat_cols):
        count = df[col].value_counts()
        ax[i].bar(count.index.to_list(), count)
        ax[i].set_title(col)
        ax[i].tick_params(axis='x', labelrotation=90)
    plt.suptitle("Categorical Columns Distribution")
    st.pyplot(plt)

    # Numeric columns
    num_cols = df.select_dtypes(include=['int', 'float']).columns
    num_rows = int(np.ceil(len(num_cols)/3))
    fig, axes = plt.subplots(num_rows, 3, layout='constrained',
                             figsize=(12, num_rows*3))
    axes = axes.flatten()
    plt.suptitle("Numerical Columns Distribution")
    for i, col in enumerate(num_cols):
        axes[i].hist(df[col])
        axes[i].set_title(col)
    st.pyplot(plt)
    fig.clear()


df = pd.read_csv("ecommerce_customer_data.csv")
print(df.info())

st.header("Data statistics before cleaning:")
display_stats(df)

df = df.drop(columns=['FavoriteCategory', 'SecondFavoriteCategory'])

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
data = df["Age"][df["Age"] >= 18]
size = df["Age"][df["Age"] < 18].count()
df.loc[df["Age"] < 18, "Age"] = np.random.choice(data, size=size, replace=True)


# Standardizing categories in columns
df.loc[df["Gender"] == 'Male', "Gender"] = 'M'
df.loc[df["Gender"] == 'Female', "Gender"] = 'F'

df.loc[df["IncomeLevel"] == 'L', "IncomeLevel"] = 'Low'
df.loc[df["IncomeLevel"] == 'H', "IncomeLevel"] = 'High'

st.header("Data statistics after cleaning:")
display_stats(df)
