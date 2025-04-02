import os
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import pickle
from typing import Dict, Any

def find_best_parameters(train_features_path: str, train_target_path: str, output_filepath: str) -> None:
    """
    Performs GridSearchCV to find the best hyperparameters for an XGBRegressor model.

    Args:
        train_features_path: Path to the training features CSV file (X_train_scaled.csv).
        train_target_path: Path to the training target CSV file (y_train.csv).
        output_filepath: Path to save the best parameters as a .pkl file.
    """
    try:
        # Load the training data
        X_train = pd.read_csv(train_features_path)
        y_train = pd.read_csv(train_target_path)

        # Define the regression model (XGBoost)
        model = XGBRegressor(random_state=42)

        # Define the parameter grid to search
        param_grid: Dict[str, list] = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

        # Perform the grid search
        print("Performing GridSearch with XGBoost...")
        grid_search.fit(X_train, y_train)

        # Get the best parameters
        best_params: Dict[str, Any] = grid_search.best_params_
        print(f"Best parameters found: {best_params}")

        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

        # Save the best parameters to a .pkl file
        with open(output_filepath, 'wb') as file:
            pickle.dump(best_params, file)

        print(f"Best parameters for XGBoost saved to '{output_filepath}'.")

    except FileNotFoundError:
        print("Error: One or both input files not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Define paths
    data_directory: str = "data"
    processed_data_subdirectory: str = "processed_data"
    models_directory: str = "models"
    train_features_filename: str = "X_train_scaled.csv"
    train_target_filename: str = "y_train.csv"
    best_params_filename: str = "best_params_xgb.pkl"  # Changed filename to indicate XGBoost

    train_features_path: str = os.path.join(data_directory, processed_data_subdirectory, train_features_filename)
    train_target_path: str = os.path.join(data_directory, processed_data_subdirectory, train_target_filename)
    output_filepath: str = os.path.join(models_directory, best_params_filename)

    # Run the function to find the best parameters for XGBoost
    find_best_parameters(train_features_path, train_target_path, output_filepath)
