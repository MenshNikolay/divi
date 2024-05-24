import pandas as pd
import matplotlib.pyplot as plt

def plot_histogram(csv_file):
    df = pd.read_csv(csv_file)

    
    total_scores = df['total_score'].values

    
    plt.figure(figsize=(10, 6))
    plt.hist(total_scores, bins=20, edgecolor='black')
    plt.xlabel('Значение totalScore')
    plt.ylabel('Частота')
    plt.title('Гистограмма значений totalScore')
    plt.grid(True)
    plt.tight_layout()

    
    plt.savefig('totalScore_histogram.png')

    
    plt.show()

if __name__ == "__main__":
    csv_file = 'result.csv'
    plot_histogram(csv_file)