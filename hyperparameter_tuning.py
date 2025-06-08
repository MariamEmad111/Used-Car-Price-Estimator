from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def tune_rf_hyperparameters(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15]
    }
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=44),
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_, grid_search.best_score_
