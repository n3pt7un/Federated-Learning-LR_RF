#!/usr/bin/env python
"""
Visualization script for Federated Learning results
---------------------------------------------------
This script generates visualizations of federated learning training progress
from the results.csv file generated during training.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.2)

# Ensure output directory exists
os.makedirs("plots", exist_ok=True)

def load_results(csv_path='results.csv'):
    """Load and preprocess results from CSV file."""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Check and fix duplicate column issue if present
        if 'round.1' in df.columns:
            # Drop the duplicate round column
            df = df.drop(columns=['round.1'])
        
        # Ensure round is treated as numeric
        df['round'] = pd.to_numeric(df['round'])
        
        # Sort by round
        df = df.sort_values('round')
        
        return df
    except Exception as e:
        print(f"Error loading results: {e}")
        return None

def plot_learning_curves(df):
    """Plot learning curves (loss and accuracy)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot loss
    ax1.plot(df['round'], df['loss'], 'o-', color='#E24A33', linewidth=2, 
             label='Training Loss')
    ax1.set_ylabel('Loss')
    ax1.set_title('Federated Learning Convergence')
    ax1.legend(loc='upper right')
    
    # Plot accuracy
    ax2.plot(df['round'], df['accuracy'], 's-', color='#348ABD', linewidth=2, 
             label='Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Training Round')
    ax2.set_ylim(0.5, 1.05)  # Appropriate for accuracy
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('plots/learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall(df):
    """Plot precision, recall, and F1 score."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df['round'], df['precision'], 'o-', color='#E24A33', linewidth=2, 
            label='Precision')
    ax.plot(df['round'], df['recall'], 's-', color='#348ABD', linewidth=2, 
            label='Recall')
    ax.plot(df['round'], df['f1_score'], '^-', color='#988ED5', linewidth=2, 
            label='F1 Score')
    
    ax.set_xlabel('Training Round')
    ax.set_ylabel('Score')
    ax.set_title('Precision, Recall, and F1 Score')
    ax.legend(loc='lower right')
    ax.set_ylim(0.5, 1.05)
    
    plt.tight_layout()
    plt.savefig('plots/precision_recall.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_rates(df):
    """Plot false positive and false negative rates."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df['round'], df['false_positive_rate'], 'o-', color='#E24A33', linewidth=2, 
            label='False Positive Rate')
    ax.plot(df['round'], df['false_negative_rate'], 's-', color='#348ABD', linewidth=2, 
            label='False Negative Rate')
    
    ax.set_xlabel('Training Round')
    ax.set_ylabel('Rate')
    ax.set_title('Error Rates')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 0.2)  # Adjust based on your expected error rates
    
    plt.tight_layout()
    plt.savefig('plots/error_rates.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_dashboard(df):
    """Create a comprehensive dashboard with all metrics."""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Loss curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['round'], df['loss'], 'o-', color='#E24A33', linewidth=2)
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    
    # Accuracy curve
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df['round'], df['accuracy'], 's-', color='#348ABD', linewidth=2)
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_ylim(0.5, 1.05)
    
    # Precision-Recall-F1
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(df['round'], df['precision'], 'o-', color='#E24A33', linewidth=2, label='Precision')
    ax3.plot(df['round'], df['recall'], 's-', color='#348ABD', linewidth=2, label='Recall')
    ax3.plot(df['round'], df['f1_score'], '^-', color='#988ED5', linewidth=2, label='F1 Score')
    ax3.set_xlabel('Training Round')
    ax3.set_ylabel('Score')
    ax3.set_title('Precision, Recall, F1 Score')
    ax3.legend(loc='lower right')
    ax3.set_ylim(0.5, 1.05)
    
    # Error rates
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(df['round'], df['false_positive_rate'], 'o-', color='#E24A33', linewidth=2, 
             label='FP Rate')
    ax4.plot(df['round'], df['false_negative_rate'], 's-', color='#348ABD', linewidth=2, 
             label='FN Rate')
    ax4.set_xlabel('Training Round')
    ax4.set_ylabel('Rate')
    ax4.set_title('Error Rates')
    ax4.legend(loc='upper right')
    ax4.set_ylim(0, 0.2)
    
    plt.tight_layout()
    plt.savefig('plots/dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_table(df):
    """Generate a summary table of the results."""
    # Extract first and last round
    first_round = df.iloc[0]
    last_round = df.iloc[-1]
    
    # Calculate improvement
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1_score', 
               'false_positive_rate', 'false_negative_rate']
    
    summary = pd.DataFrame({
        'Metric': metrics,
        'Initial Value': [first_round[m] for m in metrics],
        'Final Value': [last_round[m] for m in metrics],
        'Improvement': [last_round[m] - first_round[m] for m in metrics]
    })
    
    # Format the improvement column with an indicator of improvement direction
    def format_improvement(row):
        metric, change = row['Metric'], row['Improvement']
        # For loss and error rates, negative change is good
        if metric in ['loss', 'false_positive_rate', 'false_negative_rate']:
            return f"{change:.4f} {'✓' if change < 0 else '✗'}"
        # For others, positive change is good
        else:
            return f"{change:.4f} {'✓' if change > 0 else '✗'}"
    
    summary['Change'] = summary.apply(format_improvement, axis=1)
    
    # Save to CSV
    summary.to_csv('plots/summary.csv', index=False)
    return summary

def main():
    """Main function to generate all visualizations."""
    print("Loading federated learning results...")
    df = load_results()
    
    if df is None or df.empty:
        print("No results found or error loading results.")
        return
    
    print(f"Found {len(df)} rounds of training data.")
    
    print("Generating learning curves plot...")
    plot_learning_curves(df)
    
    print("Generating precision-recall plot...")
    plot_precision_recall(df)
    
    print("Generating error rates plot...")
    plot_error_rates(df)
    
    print("Generating comprehensive dashboard...")
    plot_dashboard(df)
    
    print("Generating summary statistics...")
    summary = generate_summary_table(df)
    print(summary)
    
    print(f"All visualizations saved to the 'plots' directory.")
    print("To view the results, open the PNG files in the 'plots' directory.")

if __name__ == "__main__":
    main() 