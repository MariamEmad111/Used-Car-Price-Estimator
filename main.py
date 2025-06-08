from data_loader import load_and_clean_data
from eda_plots import plot_brand_distribution, plot_fuel_distribution, plot_price_distribution
from preprocessing import encode_features
from data_split_scale import scale_and_split
from model_training import train_and_evaluate
from model_evaluation import plot_model_performance
from feature_importance import plot_feature_importance
from hyperparameter_tuning import tune_rf_hyperparameters

def main():
    data_path = 'Car details v3.csv'  
    
    print("Loading and cleaning data...")
    df = load_and_clean_data(data_path)
    
    print("Plotting EDA charts...")
    plot_brand_distribution(df)
    plot_fuel_distribution(df)
    plot_price_distribution(df)
    
    print("Encoding categorical features...")
    df_encoded = encode_features(df)
    
    print("Scaling and splitting the dataset...")
    X_train, X_test, y_train, y_test, scaler = scale_and_split(df_encoded)
    
    print("Training models and evaluating...")
    results_df, lr, rf, xgb = train_and_evaluate(X_train, X_test, y_train, y_test)
    print(results_df)
    
    print("Plotting model comparison graph...")
    plot_model_performance(results_df)
    
    print("Plotting feature importance...")
    feature_names = df_encoded.drop(columns=['selling_price']).columns
    plot_feature_importance(rf, feature_names)
    
    print("Hyperparameter tuning for Random Forest...")
    best_params, best_score = tune_rf_hyperparameters(X_train, y_train)
    print(f"Best Parameters: {best_params}")
    print(f"Best Cross-validation Score: {best_score * 100:.2f}%")
    
if __name__ == '__main__':
    main()
