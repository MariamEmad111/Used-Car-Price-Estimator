import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_brand_distribution(df: pd.DataFrame):
    top_brands = df['name'].value_counts().nlargest(20)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_brands.index, y=top_brands.values, palette='viridis')
    plt.xticks(rotation=45)
    plt.xlabel("Car Brand")
    plt.ylabel("Count")
    plt.title("Top 20 Car Brands")
    plt.tight_layout()
    plt.show()


def plot_fuel_distribution(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='fuel', order=df['fuel'].value_counts().index)
    plt.title("Fuel Type Distribution")
    plt.xlabel("Fuel Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_price_distribution(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    sns.histplot(df['selling_price'], bins=30, kde=True)
    plt.title("Selling Price Distribution")
    plt.xlabel("Selling Price")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_transmission_distribution(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='transmission')
    plt.title("Transmission Type Distribution")
    plt.xlabel("Transmission")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_owner_distribution(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='owner')
    plt.title("Owner Type Distribution")
    plt.xlabel("Owner Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_year_distribution(df: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x='year', bins=20, kde=True)
    plt.title("Year of Manufacture Distribution")
    plt.xlabel("Year")
    plt.ylabel("Number of Cars")
    plt.tight_layout()
    plt.show()


def plot_km_driven_distribution(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='km_driven', bins=30, kde=True)
    plt.title("Kilometers Driven Distribution")
    plt.xlabel("KM Driven")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_price_by_fuel(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='fuel', y='selling_price')
    plt.title("Selling Price by Fuel Type")
    plt.xlabel("Fuel Type")
    plt.ylabel("Selling Price")
    plt.tight_layout()
    plt.show()


def plot_price_by_transmission(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='transmission', y='selling_price')
    plt.title("Selling Price by Transmission Type")
    plt.xlabel("Transmission")
    plt.ylabel("Selling Price")
    plt.tight_layout()
    plt.show()
