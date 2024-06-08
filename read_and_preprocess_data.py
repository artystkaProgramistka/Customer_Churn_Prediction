import pandas as pd

def read_and_preprocess_data():
    # Load the data into a pandas dataframe
    data = pd.read_csv("customer_churn.csv")

    # Delete the 'Age', 'Tariff Plan', and 'Charge Amount' columns from the dataframe
    data.drop(['Age', ], axis=1, inplace=True)

    # Preprocessing
    # Ensure integer columns are treated as integers, excluding specific binary columns
    integer_columns = [column for column in data.columns if
                       column not in ['Status', 'Complains', 'Churn']]
    data[integer_columns] = data[integer_columns].astype('int')

    # Convert 'Status' to boolean according to their binary interpretation
    data['Status'] = data['Status'].map({1: True, 2: False}).astype('bool')
    data['Complains'] = data['Complains'].map({1: True, 0 : False}).astype('bool')

    # Convert 'Churn' to boolean: 1 as True (churn) and 0 as False (non-churn)
    data['Churn'] = data['Churn'].map({1: True, 0: False}).astype('bool')

    # Separate features (X) and label (y)
    X = data.drop(['Churn'], axis=1)
    y = data['Churn']

    return data, X, y