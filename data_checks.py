import pandas as pd
from read_and_preprocess_data import read_and_preprocess_data

def check_data_type_consistency(df):
    inconsistent_data = {}
    for column in df.columns:
        # Get unique data types in the column
        uniq_types = df[column].apply(type).unique()
        # If there's more than one data type, it's inconsistent
        if len(uniq_types) > 1:
            inconsistent_data[column] = uniq_types.tolist()

    # Output the inconsistent data types
    if inconsistent_data:
        for col, types in inconsistent_data.items():
            print(f"Column '{col}' has inconsistent data types: {types}")
        return inconsistent_data
    else:
        print("All columns have consistent data types.")
        return None

def check_for_missing_values(df):
    # Calculate the number of missing values per column
    missing_values = df.isnull().sum()
    # Filter out the columns that have at least one missing value
    missing_values = missing_values[missing_values > 0]
    if missing_values.empty:
        print("No missing values found.")
    else:
        # Save the missing values count to a CSV
        missing_values.to_csv('missing_values.csv')
        print(f"Missing values found: \n{missing_values}")
        print("Details saved to 'missing_values.csv'.")

def save_data_types(df):
    # Extract the data types
    data_types = df.dtypes

    # Convert the data types to a DataFrame
    data_types_df = data_types.reset_index()
    data_types_df.columns = ['Column', 'DataType']

    # Save to a CSV file
    data_types_path = 'data_types.csv'
    data_types_df.to_csv(data_types_path, index=False)
    print(f"Data types saved to '{data_types_path}'.")

def check_duplicates_and_save(df):
    # Identify all duplicates (don't mark the first occurrence as a duplicate)
    duplicate_rows = df[df.duplicated(keep=False)]

    # Check if there are any duplicates
    if not duplicate_rows.empty:
        duplicates_count = duplicate_rows.shape[0]
        print(f"Duplicate rows found: {duplicates_count}")
        # Save duplicates to a CSV file
        duplicate_rows.to_csv('duplicates.csv', index=False)
        print("Duplicates saved to 'duplicates.csv'.")
    else:
        print("No duplicate rows found.")


def mark_and_save_iqr_outliers(df, mild_threshold=1.5, extreme_threshold=3):
    try:
        numeric_df = df.select_dtypes(include=[np.number])

        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1

        # Define outlier thresholds
        mild_lower_bound = Q1 - mild_threshold * IQR
        mild_upper_bound = Q3 + mild_threshold * IQR
        extreme_lower_bound = Q1 - extreme_threshold * IQR
        extreme_upper_bound = Q3 + extreme_threshold * IQR

        # Check each numeric column against thresholds to find mild and extreme outliers
        is_mild_outlier = numeric_df.lt(mild_lower_bound) | numeric_df.gt(mild_upper_bound)
        is_extreme_outlier = numeric_df.lt(extreme_lower_bound) | numeric_df.gt(extreme_upper_bound)

        # Aggregate to identify rows where any numeric column is an outlier
        mild_outlier_mask = is_mild_outlier.any(axis=1)
        extreme_outlier_mask = is_extreme_outlier.any(axis=1)

        # Identify rows that are only mild outliers (not extreme)
        only_mild_outliers_mask = mild_outlier_mask & ~extreme_outlier_mask

        # Select full rows from the original DataFrame based on outlier masks
        only_mild_outliers_full_rows = df[only_mild_outliers_mask]
        extreme_outliers_full_rows = df[extreme_outlier_mask]

        # Save the full rows that are considered mild (but not extreme) outliers
        only_mild_outliers_full_rows.to_csv('mild_outliers.csv', index=False)
        # Save the full rows that are considered extreme outliers
        extreme_outliers_full_rows.to_csv('extreme_outliers.csv', index=False)

        # Print the number of mild (but not extreme) outliers and extreme outliers
        print(f"{len(only_mild_outliers_full_rows)} mild outliers (excluding extreme) saved to 'mild_outliers.csv'.")
        print(f"{len(extreme_outliers_full_rows)} extreme outliers saved to 'extreme_outliers.csv'.")
    except Exception as e:
        print(f"An error occurred while marking and saving IQR outliers: {e}")

def range_checks_and_save_outliers(df):
    # Define the acceptable ranges for each column
    ranges = {
        'Call Failure': (0, 50),  # Assuming no more than 50 call failures is reasonable
        'Subscription Length': (1, 100),  # Assuming the maximum subscription length of 100 months
        'Charge Amount': (0, 9),  # Already defined as ordinal 0 to 9
        'Seconds of Use': (0, 100000),  # Assuming a practical upper limit of seconds of use
        'Frequency of use': (0, 1000),  # Assuming a user won't make more than 1000 calls
        'Frequency of SMS': (0, 1000),  # Assuming a user won't send more than 1000 SMS
        'Distinct Called Numbers': (0, 500),  # Assuming a user won't call more than 500 distinct numbers
        'Age Group': (1, 5),  # Already defined range
    }

    for column, (min_val, max_val) in ranges.items():
        if column in df.columns:
            # Identify rows outside the specified range for this column
            column_outliers = df[(df[column] < min_val) | (df[column] > max_val)]

            # Check if any outliers were found and process them
            num_outliers = len(column_outliers)
            if num_outliers > 0:
                print(f"{num_outliers} outliers found in column '{column}'.")

                # Save these outliers to a CSV file named after the column
                file_path = f'{column}__range_outliers.csv'
                column_outliers.to_csv(file_path, index=False)
                print(f"Outliers for '{column}' saved to '{file_path}'.")
            else:
                print(f"No outliers found in column '{column}'.")

def perform_data_checks(df):
    save_data_types(df
    check_duplicates_and_save(df)
    mark_and_save_iqr_outliers(df)
    range_checks_and_save_outliers(df)
    check_for_missing_values(df)
    check_data_type_consistency(df)


# Use the function on your DataFrame
data, _, __ = read_and_preprocess_data()
perform_data_checks(data)