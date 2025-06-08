import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=df_importance, palette='coolwarm')
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()
