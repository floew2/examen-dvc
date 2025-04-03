import os
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from typing import Any, Dict
import json
import yaml

import mlflow

def evaluate_model(model_path: str, test_features_path: str, test_target_path: str, predictions_output_path: str, scores_output_path: str) -> None:
    """
    Loads a trained regression model, evaluates its performance, and logs metrics to MLflow.
    """
    with open("config/mlflow.yaml", "r") as f:
        mlflow_config = yaml.safe_load(f)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(mlflow_config['mlflow']['experiment_names']['evaluate'])

    with mlflow.start_run() as run:
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("test_features_path", test_features_path)
        mlflow.log_param("test_target_path", test_target_path)

        try:
            # Load the trained model
            with open(model_path, 'rb') as file:
                model: Any = pickle.load(file)

            print(f"Loaded trained model from '{model_path}'.")

            # Load the test data
            X_test = pd.read_csv(test_features_path)
            y_test = pd.read_csv(test_target_path)

            # Make predictions on the test set
            print("Making predictions on the test set...")
            predictions = model.predict(X_test)

            # Save the predictions to a CSV file
            predictions_df = pd.DataFrame({'predicted_silica_concentrate': predictions})
            os.makedirs(os.path.dirname(predictions_output_path), exist_ok=True)
            predictions_df.to_csv(predictions_output_path, index=False)
            print(f"Predictions saved to '{predictions_output_path}'.")
            mlflow.log_artifact(predictions_output_path, "predictions.csv")

            # Evaluate the model
            mse: float = mean_squared_error(y_test, predictions)
            r2: float = r2_score(y_test, predictions)

            # Log the evaluation metrics
            mlflow.log_metrics({"mean_squared_error": mse, "r2_score": r2})
            print(f"Evaluation metrics: MSE={mse:.4f}, R2={r2:.4f}")

            # Create the metrics directory if it doesn't exist
            metrics_directory: str = os.path.dirname(scores_output_path)
            os.makedirs(metrics_directory, exist_ok=True)

            # Save the evaluation metrics to a JSON file (optional, but good practice)
            scores: Dict[str, float] = {
                "mean_squared_error": mse,
                "r2_score": r2
            }
            with open(scores_output_path, 'w') as file:
                json.dump(scores, file, indent=4)
            mlflow.log_artifact(scores_output_path, "scores.json")

            print(f"Evaluation metrics saved to '{scores_output_path}'.")

        except FileNotFoundError:
            print("Error: One or more input files not found.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Define paths (same as before)
    data_directory: str = "data"
    processed_data_subdirectory: str = "processed_data"
    models_directory: str = "models"
    reports_directory: str = "metrics"
    test_features_filename: str = "X_test_scaled.csv"
    test_target_filename: str = "y_test.csv"
    trained_model_filename: str = "trained_model_xgb.pkl"
    predictions_filename: str = "predictions.csv"
    scores_filename: str = "scores.json"

    model_path: str = os.path.join(models_directory, trained_model_filename)
    test_features_path: str = os.path.join(data_directory, processed_data_subdirectory, test_features_filename)
    test_target_path: str = os.path.join(data_directory, processed_data_subdirectory, test_target_filename)
    predictions_output_path: str = os.path.join(data_directory, predictions_filename)
    scores_output_path: str = os.path.join(reports_directory, scores_filename)

    # Run the model evaluation function
    evaluate_model(model_path, test_features_path, test_target_path, predictions_output_path, scores_output_path)
