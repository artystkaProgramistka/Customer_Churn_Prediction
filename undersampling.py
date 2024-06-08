import pandas as pd
from sklearn.utils import resample

# Wczytanie danych z pliku CSV
df = pd.read_csv('customer_churn_distinct_without_outliers.csv')

# Sprawdzenie rozkładu klas
class_counts = df['Churn'].value_counts()
print(class_counts)

# Rozdzielenie danych na klasy większościową i mniejszościową
df_majority = df[df['Churn'] == class_counts.idxmax()]
df_minority = df[df['Churn'] == class_counts.idxmin()]

# Undersampling klasy większościowej
df_majority_undersampled = resample(df_majority, 
                                    replace=False,    # bez zamiany
                                    n_samples=len(df_minority),   # liczba próbek jak w klasie mniejszościowej
                                    random_state=123) # dla powtarzalności

# Połączenie klas po undersamplingu
df_undersampled = pd.concat([df_majority_undersampled, df_minority])

# Sprawdzenie nowego rozkładu klas
new_class_counts = df_undersampled['Churn'].value_counts()
print(new_class_counts)

# Zapisanie przetworzonych danych do nowego pliku CSV
df_undersampled.to_csv('customer_churn_undersampled.csv', index=False)

