from read_and_preprocess_data import read_and_preprocess_data
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def save_descriptive_statistics(df):
    try:
        df.describe().to_csv('descriptive_stats.csv')
        print("Descriptive statistics saved to 'descriptive_stats.csv'.")
    except Exception as e:
        print(f"An error occurred while saving descriptive statistics: {e}")

# Assuming read_and_preprocess_data is properly defined in your read_and_preprocess_data.py file
from read_and_preprocess_data import read_and_preprocess_data

import matplotlib.pyplot as plt
import pandas as pd

# Assuming read_and_preprocess_data is properly defined in your read_and_preprocess_data.py file
from read_and_preprocess_data import read_and_preprocess_data

def generate_and_save_plots(df):
    """
    Generates histograms and box plots for all numerical columns and histograms for binary columns in the DataFrame.
    Each type of plot is saved as a separate PNG file.
    """
    # Convert boolean columns to integers for histogram purposes
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)  # Convert bool to int

    # Filter for numeric columns in the DataFrame (including converted boolean columns)
    cols = df.select_dtypes(include=['number']).columns

    # Determine the number of rows needed for subplots based on the number of columns
    num_plots_hist = len(cols)
    nrows_hist = (num_plots_hist // 3) + (1 if num_plots_hist % 3 else 0)

    # Create histograms for all numeric (including converted boolean) columns
    fig_hist, axs_hist = plt.subplots(nrows=nrows_hist, ncols=3, figsize=(15, 5 * nrows_hist))
    axs_hist = axs_hist.flatten()  # Flatten to 1D array for easy iteration

    for i, col in enumerate(cols):
        df[col].hist(ax=axs_hist[i], bins=20)
        axs_hist[i].set_title(f'Histogram of {col}')
        axs_hist[i].set_xlabel('Value')
        axs_hist[i].set_ylabel('Frequency')

    # Remove empty subplots if the number of numeric columns isn't a multiple of 3 for histograms
    for j in range(i + 1, len(axs_hist)):
        fig_hist.delaxes(axs_hist[j])

    plt.tight_layout()
    plt.savefig('histograms.png')
    plt.close(fig_hist)
    print("Histograms saved to 'histograms.png'.")

    # Create box plots for all numeric columns
    num_plots_box = len(cols)
    nrows_box = (num_plots_box // 3) + (1 if num_plots_box % 3 else 0)

    fig_box, axs_box = plt.subplots(nrows=nrows_box, ncols=3, figsize=(15, 5 * nrows_box))
    axs_box = axs_box.flatten()  # Flatten to 1D array for easy iteration

    for i, col in enumerate(cols):
        df.boxplot(column=[col], ax=axs_box[i])
        axs_box[i].set_title(f'Box Plot of {col}')

    # Remove empty subplots if the number of numeric columns isn't a multiple of 3 for box plots
    for j in range(i + 1, len(axs_box)):
        fig_box.delaxes(axs_box[j])

    plt.tight_layout()
    plt.savefig('box_plots.png')
    plt.close(fig_box)
    print("Box plots saved to 'box_plots.png'.")

    # Generate and save the correlation matrix
    corr_matrix = df[cols].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Correlation coefficient'})
    plt.savefig('correlation_matrix.png')
    plt.close()
    print("Correlation matrix saved to 'correlation_matrix.png'.")

def perform_pca_analysis(df):
    # Convert boolean columns to integers for histogram purposes
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)  # Convert bool to int

    # Filter for numeric columns in the DataFrame (including converted boolean columns)
    cols = df.select_dtypes(include=['number']).columns
    data_numeric = df[cols]

    # Standardizing the data to have a mean of 0 and a variance of 1
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_numeric)

    pca = PCA()
    pca.fit(data_scaled)

    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
    plt.title('Explained Variance by Comoponents')
    plt.xlabel('Number of Conponents')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.savefig('pca_explained_variance.png')
    plt.close()
    print("PCA analysis and screen plot saved to 'pca_explained_variance.png'.")
    return pca

# Example usage
if __name__ == "__main__":
    data, _, __ = read_and_preprocess_data()  # Load data
    print(data.dtypes)
    generate_and_save_plots(data)  # Generate and save plots
    pca_model = perform_pca_analysis(data)




