import os
import pandas as pd
import pickle
from xgboost import XGBRegressor
from typing import Dict, Any

def train_model(train_features_path: str, train_target_path: str, best_params_path: str, output_filepath: str) -> None:
    """
    Trains an XGBRegressor model using the best parameters found by GridSearchCV
    and saves the trained model to a .pkl file.

    Args:
        train_features_path: Path to the scaled training features CSV file (X_train_scaled.csv).
        train_target_path: Path to the training target CSV file (y_train.csv).
        best_params_path: Path to the .pkl file containing the best parameters for XGBoost.
        output_filepath: Path to save the trained XGBoost model as a .pkl file.
    """
    try:
        # Load the training data
        X_train = pd.read_csv(train_features_path)
        y_train = pd.read_csv(train_target_path)

        # Load the best parameters from the .pkl file
        with open(best_params_path, 'rb') as file:
            best_params: Dict[str, Any] = pickle.load(file)

        print(f"Loaded best parameters for XGBoost: {best_params}")

        # Initialize the regression model with the best parameters (XGBoost)
        model = XGBRegressor(**best_params, random_state=42)

        # Train the model
        print("Training the XGBoost model...")
        model.fit(X_train, y_train)

        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

        # Save the trained model to a .pkl file
        with open(output_filepath, 'wb') as file:
            pickle.dump(model, file)

        print(f"Trained XGBoost model saved to '{output_filepath}'.")

    except FileNotFoundError:
        print("Error: One or more input files not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Define paths
    data_directory: str = "data"
    processed_data_subdirectory: str = "processed_data"
    models_directory: str = "models"
    train_features_filename: str = "X_train_scaled.csv"
    train_target_filename: str = "y_train.csv"
    best_params_filename: str = "best_params_xgb.pkl"
    trained_model_filename: str = "trained_model_xgb.pkl"

    train_features_path: str = os.path.join(data_directory, processed_data_subdirectory, train_features_filename)
    train_target_path: str = os.path.join(data_directory, processed_data_subdirectory, train_target_filename)
    best_params_path: str = os.path.join(models_directory, best_params_filename)
    output_filepath: str = os.path.join(models_directory, trained_model_filename)

    # Run the function to train the XGBoost model
    train_model(train_features_path, train_target_path, best_params_path, output_filepath)
