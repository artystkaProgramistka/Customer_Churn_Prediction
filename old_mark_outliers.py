def mark_outliers(df, threshold=1.5):
    outliers = pd.DataFrame(columns=df.columns)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (threshold * IQR)
            upper_bound = Q3 + (threshold * IQR)

            # Filter out the outliers from the dataframe
            col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

            # Mark the outliers in the new dataframe
            outliers = pd.concat([outliers, col_outliers], axis=0)

    # Drop duplicates since a row may contain more than one outlier
    outliers = outliers.drop_duplicates()
    return outliers