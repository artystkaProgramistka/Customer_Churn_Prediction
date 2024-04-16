from read_and_preprocess_data import read_and_preprocess_data
import matplotlib.pyplot as plt
import pandas as pd

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
    Generates histograms and box plots for all numerical columns in the DataFrame.
    Each type of plot is saved as a separate PNG file.
    """
    # Filter for numeric columns in the DataFrame
    numeric_cols = df.select_dtypes(include=['number']).columns

    # Determine the number of rows needed for subplots based on the number of numeric columns
    # Arrange plots in a 3-column format for both types of plots
    num_plots = len(numeric_cols)
    nrows = (num_plots // 3) + (1 if num_plots % 3 else 0)

    # Create histograms
    fig_hist, axs_hist = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 5 * nrows))
    axs_hist = axs_hist.flatten()  # Flatten to 1D array for easy iteration

    for i, col in enumerate(numeric_cols):
        df[col].hist(ax=axs_hist[i], bins=20)
        axs_hist[i].set_title(f'Histogram of {col}')
        axs_hist[i].set_xlabel('Value')
        axs_hist[i].set_ylabel('Frequency')

    # Remove empty subplots if the number of numeric columns isn't a multiple of 3 for histograms
    for j in range(i + 1, len(axs_hist)):
        fig_hist.delaxes(axs_hist[j])

    plt.tight_layout()
    plt.savefig('histograms.png')
    plt.close(fig_hist)  # Close the histogram plotting window
    print("Histograms saved to 'histograms.png'.")

    # Create box plots
    fig_box, axs_box = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 5 * nrows))
    axs_box = axs_box.flatten()  # Flatten to 1D array for easy iteration

    for i, col in enumerate(numeric_cols):
        df.boxplot(column=[col], ax=axs_box[i])
        axs_box[i].set_title(f'Box Plot of {col}')

    # Remove empty subplots if the number of numeric columns isn't a multiple of 3 for box plots
    for j in range(i + 1, len(axs_box)):
        fig_box.delaxes(axs_box[j])

    plt.tight_layout()
    plt.savefig('box_plots.png')
    plt.close(fig_box)  # Close the box plot plotting window
    print("Box plots saved to 'box_plots.png'.")

# Example usage
if __name__ == "__main__":
    data, _, __ = read_and_preprocess_data()  # Load your data
    generate_and_save_plots(data)  # Generate and save plots


# Example usage
if __name__ == "__main__":
    data, _, __ = read_and_preprocess_data()  # Load your data
    generate_and_save_plots(data)  # Generate and save histograms




