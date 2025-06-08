from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import pandas as pd

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    # Label encode car brand name
    le = LabelEncoder()
    df['name'] = le.fit_transform(df['name'])
    
    # Ordinal encode owner with logical order
    owner_order = [['Test Drive Car', 'First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner']]
    oe = OrdinalEncoder(categories=owner_order)
    df['owner'] = oe.fit_transform(df[['owner']])
    
    # One-hot encode categorical columns
    cat_cols = ['fuel', 'seller_type', 'transmission']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Convert all to int if possible
    df = df.astype(int)
    
    return df
