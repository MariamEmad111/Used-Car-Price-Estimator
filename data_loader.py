import pandas as pd
import os

def load_car_data(filename: str = 'Car details v3.csv') -> pd.DataFrame:
    """
    Load car details data from a CSV file.

    Parameters:
    filename (str): Path to the CSV file. Default is 'Car details v3.csv'.

    Returns:
    pd.DataFrame: DataFrame containing car details.
    """
    # Get absolute path relative to this script file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_dir, filename)
    
    try:
        data = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}")
        return data
    except FileNotFoundError:
        print(f"Error: The file {filepath} does not exist.")
        raise
    except pd.errors.EmptyDataError:
        print(f"Error: The file {filepath} is empty.")
        raise
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        raise

if __name__ == "__main__":
    # When running this script directly, load the data and show a summary
    df = load_car_data()
    print(df.head())
    print(f"Total records: {len(df)}")
