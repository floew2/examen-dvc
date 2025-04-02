import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def normalize_data(train_filepath: str, test_filepath: str, output_dir: str) -> None:
    """
    Normalizes the training and testing datasets using StandardScaler and saves the scaled data as CSV files.

    Args:
        train_filepath: Path to the training features CSV file (X_train.csv).
        test_filepath: Path to the testing features CSV file (X_test.csv).
        output_dir: Directory to save the scaled datasets.
    """
    try:
        # Load the training and testing datasets
        X_train = pd.read_csv(train_filepath)
        X_test = pd.read_csv(test_filepath)

        # Initialize the StandardScaler
        scaler = StandardScaler()

        # Fit the scaler on the training data and transform it
        X_train_scaled = scaler.fit_transform(X_train)

        # Transform the testing data using the fitted scaler
        X_test_scaled = scaler.transform(X_test)

        # Convert the scaled arrays back to pandas DataFrames
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Define the output file paths
        X_train_scaled_path = os.path.join(output_dir, "X_train_scaled.csv")
        X_test_scaled_path = os.path.join(output_dir, "X_test_scaled.csv")

        # Function to save dataframes, overwriting if exists
        def save_dataframe(df_to_save: pd.DataFrame, filepath: str) -> None:
            """Saves a pandas DataFrame to a CSV file, overwriting if the file exists."""
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"Existing file found at '{filepath}'. Overwriting.")
            df_to_save.to_csv(filepath, index=False)
            print(f"Saved data to '{filepath}'.")

        # Save the scaled datasets
        save_dataframe(X_train_scaled_df, X_train_scaled_path)
        save_dataframe(X_test_scaled_df, X_test_scaled_path)

        print("Data normalization complete.")

    except FileNotFoundError:
        print(f"Error: One or both input files not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Define paths
    data_directory: str = "data"
    processed_data_subdirectory: str = "processed_data"
    output_directory: str = os.path.join(data_directory, processed_data_subdirectory)
    train_features_filename: str = "X_train.csv"
    test_features_filename: str = "X_test.csv"

    train_filepath: str = os.path.join(output_directory, train_features_filename)
    test_filepath: str = os.path.join(output_directory, test_features_filename)

    # Run the data normalization function
    normalize_data(train_filepath, test_filepath, output_directory)
