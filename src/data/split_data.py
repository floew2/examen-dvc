import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def split_data(input_filepath: str, output_dir: str, target_column: str, test_size: float = 0.2, random_state: int = 42) -> None:
    """
    Loads, preprocesses (removes 'date' column, handles duplicates and missing values),
    splits the input data into training and testing sets, and saves them as CSV files.

    Args:
        input_filepath: Path to the input CSV file.
        output_dir: Directory to save the train and test datasets.
        target_column: Name of the target variable column.
        test_size: Proportion of the dataset to use for the test set (default: 0.2).
        random_state: Random seed for reproducibility (default: 42).
    """
    try:
        # Load the dataset
        df = pd.read_csv(input_filepath)
        initial_rows = len(df)
        print(f"Loaded dataset with {initial_rows} rows.")

        # Remove the 'date' column if it exists
        if 'date' in df.columns:
            df = df.drop(columns=['date'])
            print("Removed 'date' column.")
        else:
            print("Warning: 'date' column not found in the dataset.")

        # Handle duplicates
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            df = df.drop_duplicates().reset_index(drop=True)
            print(f"Removed {duplicate_rows} duplicate rows. Remaining rows: {len(df)}.")
        else:
            print("No duplicate rows found.")

        # Handle missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            rows_before_dropna = len(df)
            df = df.dropna().reset_index(drop=True)
            rows_dropped = rows_before_dropna - len(df)
            print(f"Removed {rows_dropped} rows with missing values. Remaining rows: {len(df)}.")
        else:
            print("No missing values found.")

        # Separate features (X) and target (y)
        y = df[target_column]
        X = df.drop(columns=[target_column])

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Define the output file paths
        X_train_path = os.path.join(output_dir, "X_train.csv")
        X_test_path = os.path.join(output_dir, "X_test.csv")
        y_train_path = os.path.join(output_dir, "y_train.csv")
        y_test_path = os.path.join(output_dir, "y_test.csv")

        # Function to save dataframes, overwriting if exists
        def save_dataframe(df_to_save: pd.DataFrame, filepath: str) -> None:
            """Saves a pandas DataFrame to a CSV file, overwriting if the file exists."""
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"Existing file found at '{filepath}'. Overwriting.")
            df_to_save.to_csv(filepath, index=False)
            print(f"Saved data to '{filepath}'.")

        # Save the datasets
        save_dataframe(X_train, X_train_path)
        save_dataframe(X_test, X_test_path)
        save_dataframe(y_train, y_train_path)
        save_dataframe(y_test, y_test_path)

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_filepath}'.")
    except KeyError:
        print(f"Error: Target column '{target_column}' not found in the dataset.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Define paths and parameters
    data_directory: str = "data"
    raw_data_subdirectory: str = "raw_data"
    processed_data_subdirectory: str = "processed_data"
    input_filename: str = "raw.csv"
    target_column_name: str = "silica_concentrate"
    output_directory: str = os.path.join(data_directory, processed_data_subdirectory)
    input_filepath: str = os.path.join(data_directory, raw_data_subdirectory, input_filename)

    # Run the data splitting function
    split_data(input_filepath, output_directory, target_column_name)
