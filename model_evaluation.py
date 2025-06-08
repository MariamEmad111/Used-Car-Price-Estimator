import matplotlib.pyplot as plt

def plot_model_performance(results_df):
    plt.figure(figsize=(8, 5))
    plt.plot(results_df['Model'], results_df['Test Accuracy (%)'], marker='o', linestyle='-', color='red')
    plt.title("Model Test Accuracy Comparison")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
