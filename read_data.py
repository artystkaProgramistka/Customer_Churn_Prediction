import pandas as pd
import numpy as np

def read_and_preprocess_data():
    # Load the data into a pandas dataframe
    data = pd.read_csv("customer_churn.csv")

    y = data['Churn'].astype('bool')
    X = data.drop(['Churn'], axis=1)

    # preprocessing
    integer_columns = [column for column in X.columns if column not in ['Status', 'Complains']]
    X[integer_columns] = X[integer_columns].astype('int')
    data[integer_columns] = X[integer_columns].astype('int')
    binary_columns = [column for column in X.columns if column not in integer_columns]
    X[binary_columns] = X[binary_columns].astype('bool')
    data[binary_columns] = X[binary_columns].astype('bool')
    data['Churn'] = data['Churn'].astype('bool')

    return data, X, y

def mark_outliers(df, threshold=1.5):
    outliers = pd.DataFrame(columns=df.columns)
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (threshold * IQR)
        upper_bound = Q3 + (threshold * IQR)
        col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers = pd.concat([outliers, col_outliers], axis=0)
    outliers = outliers.drop_duplicates()
    return outliers

def perform_data_checks(df):
    try:
        # Check for duplicate data
        duplicates = df.duplicated().sum()
        print(f"Duplicate rows found: {duplicates}" if duplicates else "No duplicate rows found.")
    except Exception as e:
        print(f"An error occurred while checking for duplicates: {e}")

    try:
        # Check for consistency in categorical data
        for column in df.select_dtypes(include=['object']).columns:
            unique_values = df[column].unique()
            print(f"{column} unique values: {unique_values}")
    except Exception as e:
        print(f"An error occurred while checking categorical data consistency: {e}")

    try:
        # Check data types
        print("Data types:\n", df.dtypes)
    except Exception as e:
        print(f"An error occurred while checking data types: {e}")

    try:
        # Check for unrealistic or impossible values
        df.describe().to_csv('descriptive_stats.csv')
    except Exception as e:
        print(f"An error occurred while describing data: {e}")

    try:
        # Check for consistent syntax in object columns
        for column in df.select_dtypes(include=['object']).columns:
            unique_values_pre = df[column].unique()
            df[column] = df[column].str.lower().str.strip()
            unique_values_post = df[column].unique()
            if not np.array_equal(unique_values_pre, unique_values_post):
                print(f"Syntax inconsistencies found and fixed in column '{column}'.")
    except Exception as e:
        print(f"An error occurred while checking syntax consistency: {e}")

    numeric_df = df.select_dtypes(include=[np.number])
    try:
        # Calculate quartiles and IQR once
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1

        # Mark outliers within 1.5 IQR
        mild_outlier_mask = (numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))
        mild_outliers = mild_outlier_mask.sum().sum()
        # Save the mild outliers
        mild_outliers_df = numeric_df[mild_outlier_mask]
        mild_outliers_df.to_csv('outliers.csv', index=False)
        print(
            f"Outliers larger than 1.5 IQR found: {mild_outliers}" if mild_outliers else "No outliers larger than 1.5 IQR found.")

        # Mark outliers larger than 3 IQR using the same IQR calculation
        extreme_outlier_mask = (numeric_df < (Q1 - 3 * IQR)) | (numeric_df > (Q3 + 3 * IQR))
        extreme_outliers = extreme_outlier_mask.sum().sum()
        # Save the extreme outliers
        extreme_outliers_df = numeric_df[extreme_outlier_mask]
        extreme_outliers_df.to_csv('extreme_outliers.csv', index=False)
        print(
            f"Outliers larger than 3 IQR found: {extreme_outliers}" if extreme_outliers else "No outliers larger than 3 IQR found.")
    except Exception as e:
        print(f"An error occurred while marking IQR outliers: {e}")


# Use the function on your DataFrame
data, _, __ = read_and_preprocess_data()
perform_data_checks(data)