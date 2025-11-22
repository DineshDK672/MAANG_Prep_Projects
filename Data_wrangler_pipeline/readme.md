# Data Wrangler Pipeline

## Project Overview

This is a Python-based data wrangling pipeline integrated with a Streamlit dashboard. It processes and cleans an e-commerce customer dataset, showcasing key statistics before and after cleaning. The project aims to visualize data quality (null values, categorical and numerical distributions) interactively while demonstrating practical data cleaning techniques such as null value imputation, type conversion, and category standardization.

## Features

- Displays null value distribution across columns.

- Visualizes categorical and numerical column distributions using matplotlib plots in Streamlit.

- Cleans customer engagement data by:

  - Trimming and cleaning customer IDs.

  - Imputing null values with random samples from existing data.

  - Handling age outliers using random imputation.

  - Standardizing categorical variables (e.g., gender, income level).

- Uses Streamlit for an interactive and user-friendly dashboard interface.

## Technologies Used

- **Python 3**

- **Pandas** for data handling and cleaning

- **NumPy** for numerical operations and random sampling

- **Matplotlib** for data visualization

- **Streamlit** for building the interactive dashboard
