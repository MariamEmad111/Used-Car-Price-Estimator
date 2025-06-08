from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pandas as pd
import joblib

def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = []
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_score = lr.score(X_test, y_test) * 100
    results.append(("Linear Regression", lr_score))
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=44)
    rf.fit(X_train, y_train)
    rf_score = rf.score(X_test, y_test) * 100
    results.append(("Random Forest", rf_score))
    
    # Save Random Forest model
    joblib.dump(rf, 'random_forest_model.pkl')
    
    # XGBoost
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=44)
    xgb.fit(X_train, y_train)
    xgb_score = xgb.score(X_test, y_test) * 100
    results.append(("XGBoost", xgb_score))
    
    results_df = pd.DataFrame(results, columns=['Model', 'Test Accuracy (%)'])
    return results_df, lr, rf, xgb
