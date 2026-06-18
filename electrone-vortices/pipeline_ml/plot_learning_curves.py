"""
Script to visualize the learning curves of the MLP model.
Plots Epoch vs MSE and Epoch vs R^2 score with academic styling.
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_learning_curves(metrics_file='../data/data_learning/training_metrics.csv'):
    if not os.path.exists(metrics_file):
        print(f"Error: File '{metrics_file}' not found. Run main.py first to train the model.")
        return

    epochs, train_mse, val_mse, val_r2 = [], [], [], []

    # Wczytanie logów z treningu
    with open(metrics_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_mse.append(float(row['train_mse']))
            val_mse.append(float(row['val_mse']))
            val_r2.append(float(row['val_r2']))

    # Konwersja na arraye numpy dla łatwiejszego manipulowania
    epochs = np.array(epochs)
    train_mse = np.array(train_mse)
    val_mse = np.array(val_mse)
    val_r2 = np.array(val_r2)

    # Inicjalizacja wykresu: 1 wiersz, 2 kolumny
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ==========================================
    # Subplot 1: Epoch vs MSE Loss
    # ==========================================
    ax1.plot(epochs, train_mse, color='#1F77B4', linewidth=2.0, label='Train MSE')
    ax1.plot(epochs, val_mse, color='#B23B3B', linewidth=2.0, linestyle='--', label='Validation MSE')

    ax1.set_title('Empirical Risk Minimization (MSE)', fontsize=14, pad=10)
    ax1.set_xlabel('Epoch', fontsize=14, family='sans-serif')
    ax1.set_ylabel('Mean Squared Error', fontsize=14, family='sans-serif')

    # Skala logarytmiczna dla MSE często lepiej obrazuje proces zbiegania
    ax1.set_yscale('log')
    ax1.set_xlim([0, max(epochs)])
    ax1.legend(fontsize=12, loc='upper right')

    # ==========================================
    # Subplot 2: Epoch vs R^2 Score
    # ==========================================
    ax2.plot(epochs, val_r2, color='#31A354', linewidth=2.0, label=r'Validation $R^2$')

    ax2.set_title(r'Coefficient of Determination ($R^2$)', fontsize=14, pad=10)
    ax2.set_xlabel('Epoch', fontsize=14, family='sans-serif')
    ax2.set_ylabel(r'$R^2$ Score', fontsize=14, family='sans-serif')

    ax2.set_xlim([0, max(epochs)])
    # Ustawienie limitów osi Y, ucinamy głębokie wartości ujemne na początku uczenia dla czytelności
    ax2.set_ylim([max(-1.0, min(val_r2)), 1.05])
    ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)  # Referencyjna linia perfekcji
    ax2.legend(fontsize=12, loc='lower right')

    # ==========================================
    # Academic styling applied to both subplots
    # ==========================================
    for ax in [ax1, ax2]:
        ax.tick_params(direction='in', top=True, right=True, which='both', labelsize=12, width=1.0)
        ax.grid(True, linestyle=':', alpha=0.5)

        # Ramki w stylu naukowym
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_color('black')
            ax.spines[spine].set_linewidth(1.0)

    plt.tight_layout()

    # Zapis obrazu
    output_dir = Path('../data/data_learning/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_image = output_dir / 'mlp_learning_curves.png'

    plt.savefig(output_image, dpi=400)
    print(f"Plot successfully generated and saved as: {output_image}")
    plt.show()


if __name__ == '__main__':
    plot_learning_curves()