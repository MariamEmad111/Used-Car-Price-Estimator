from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def scale_and_split(df, target_col='selling_price', test_size=0.25, random_state=44):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    return X_train, X_test, y_train, y_test, scaler
