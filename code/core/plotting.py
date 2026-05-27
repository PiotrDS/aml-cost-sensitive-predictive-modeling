import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set a consistent beautiful style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")

def plot_feature_importance(features, importances, title, output_dir, filename="feature_importance.png", top_n=20):
    """Consistent bar plot for feature importances/coefficients."""
    if len(features) == 0:
        return
    
    # Take top N
    idx = np.argsort(np.abs(importances))[::-1][:top_n]
    sel_features = [features[i] for i in idx]
    sel_importances = [importances[i] for i in idx]
    
    plt.figure(figsize=(10, max(6, int(len(sel_features) * 0.3))))
    
    # Colored based on positive/negative if applicable (e.g. Lasso)
    colors = ['#2ca02c' if v >= 0 else '#d62728' for v in sel_importances]
    
    y_pos = np.arange(len(sel_features))
    plt.barh(y_pos, sel_importances, color=colors, edgecolor='black', alpha=0.8)
    plt.yticks(y_pos, sel_features)
    plt.gca().invert_yaxis()  # Highest at top
    
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Importance / Coefficient', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def plot_probability_distribution(y_true, probs, title, output_dir, filename="prob_distribution.png"):
    """Consistent KDE plot for probability distributions of classes."""
    plt.figure(figsize=(10, 6))
    
    probs_pos = probs[y_true == 1]
    probs_neg = probs[y_true == 0]
    
    sns.kdeplot(probs_neg, fill=True, color='#d62728', label='Class 0 (No Conversion)', alpha=0.4)
    sns.kdeplot(probs_pos, fill=True, color='#2ca02c', label='Class 1 (Conversion)', alpha=0.4)
    
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Predicted Probability P(Y=1)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(loc='upper right')
    plt.xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def plot_profit_optimization_curve(x_values, profit_values, x_label, title, best_x, output_dir, filename="profit_optimization.png"):
    """Consistent curve plot for profit optimization."""
    plt.figure(figsize=(10, 6))
    
    plt.plot(x_values, profit_values, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=6)
    plt.axvline(x=best_x, color='red', linestyle='--', linewidth=2, label=f'Best: {best_x}')
    
    best_profit = max(profit_values)
    plt.axhline(y=best_profit, color='green', linestyle=':', linewidth=1.5, alpha=0.6, label=f'Max Profit: {best_profit:.1f} EUR')
    
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Estimated OOF Profit (EUR)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def plot_lasso_sweep(c_values, profit_values, features_count, best_c, output_dir, filename="lasso_sweep.png"):
    """Specific 2-subplot graph for Lasso."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(c_values, profit_values, marker="o", color="#1f77b4", lw=2, markersize=5)
    ax1.axvline(best_c, color="red", ls="--", label=f"Best C={best_c:.4f}")
    ax1.set_ylabel("CV Profit (EUR)", fontsize=11)
    ax1.set_title("Profit vs. L1 Regularization Strength (Nested MI)", fontsize=13, pad=10)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(c_values, features_count, marker="s", color="#2ca02c", lw=2, markersize=5)
    ax2.axvline(best_c, color="red", ls="--")
    ax2.set_xlabel("C (Inverse Regularization Strength)", fontsize=11)
    ax2.set_ylabel("Avg Features per Fold", fontsize=11)
    ax2.set_xscale("log")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
