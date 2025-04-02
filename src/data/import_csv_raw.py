import os
import requests
import shutil
from typing import Optional

def download_raw_data(data_url: str, output_path: str) -> None:
    """
    Downloads raw data from a given URL and saves it to the specified output path.

    Args:
        data_url: The URL of the CSV data file.
        output_path: The full path to the output CSV file.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Check if the file already exists
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Existing file found at '{output_path}'. Deleted and will be overwritten.")

    try:
        # Download the data
        response = requests.get(data_url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Save the data to the output file
        with open(output_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"Raw data downloaded successfully to '{output_path}'.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading data from '{data_url}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    data_directory: str = "data"
    raw_data_subdirectory: str = "raw_data"
    output_directory: str = os.path.join(data_directory, raw_data_subdirectory)
    output_filename: str = "raw.csv"
    output_file_path: str = os.path.join(output_directory, output_filename)
    raw_data_url: str = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"

    download_raw_data(raw_data_url, output_file_path)
