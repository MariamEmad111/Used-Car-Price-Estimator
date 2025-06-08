import pandas as pd
import numpy as np
import os

def load_and_clean_car_data(filename: str = 'Car details v3.csv') -> pd.DataFrame:
    """
    Load and clean car details data from a CSV file.

    Cleaning steps:
    - Handle missing values by dropping or filling.
    - Convert year to age of the car (assuming current year).
    - Clean numeric columns by coercing errors to NaN and filling.
    - Strip whitespace from string columns.
    
    Parameters:
    filename (str): Path to the CSV file. Default is 'Car details v3.csv'.

    Returns:
    pd.DataFrame: Cleaned car details data.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_dir, filename)

    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}")

        # Strip whitespace from string columns to avoid trailing spaces
        str_cols = df.select_dtypes(include='object').columns
        df[str_cols] = df[str_cols].apply(lambda x: x.str.strip())

        # Handling missing values
        # Option 1: drop rows with any missing data (if dataset is big)
        # df = df.dropna()

        # Option 2: fill missing numeric columns with median, categorical with mode
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                median_val = df[col].median()
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col].fillna(median_val, inplace=True)
            else:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col].fillna(mode_val[0], inplace=True)

        # Convert 'Year' column to 'Car_Age' assuming current year is 2025
        if 'Year' in df.columns:
            current_year = 2025
            df['Car_Age'] = current_year - pd.to_numeric(df['Year'], errors='coerce')
            # If any 'Year' was invalid, 'Car_Age' will be NaN - fill with median age
            median_age = df['Car_Age'].median()
            df['Car_Age'].fillna(median_age, inplace=True)
            # Drop original 'Year' column if you want
            df.drop(columns=['Year'], inplace=True)

        # Clean numeric columns again after conversions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)

        print("Data cleaning complete.")
        return df

    except FileNotFoundError:
        print(f"Error: The file {filepath} does not exist.")
        raise
    except pd.errors.EmptyDataError:
        print(f"Error: The file {filepath} is empty.")
        raise
    except Exception as e:
        print(f"An error occurred while loading or cleaning the data: {e}")
        raise

if __name__ == "__main__":
    df = load_and_clean_car_data()
    print(df.head())
    print(f"Total records after cleaning: {len(df)}")
