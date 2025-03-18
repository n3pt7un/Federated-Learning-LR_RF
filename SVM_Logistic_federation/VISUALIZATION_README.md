# Federated Learning Results Visualization

This set of scripts helps you visualize and analyze the results of your federated learning experiments.

## Quick Start

1. Install dependencies:
   ```
   python install_viz_deps.py
   ```

2. Run the visualization script:
   ```
   python visualize_results.py
   ```

3. Check the generated visualizations in the `plots` directory.

## Generated Visualizations

The visualization script generates several plots to help you analyze your federated learning model performance:

1. **Learning Curves** (`learning_curves.png`): Shows the progression of loss and accuracy over training rounds.

2. **Precision-Recall** (`precision_recall.png`): Displays precision, recall, and F1 score trends over rounds.

3. **Error Rates** (`error_rates.png`): Shows false positive and false negative rates over time.

4. **Dashboard** (`dashboard.png`): A comprehensive view of all key metrics in a single image.

5. **Summary Statistics** (`summary.csv`): A table showing initial values, final values, and improvements for all metrics.

## Understanding the Results

### Accuracy, Precision, Recall, F1-Score
- Higher values indicate better performance (closer to 1.0 is better)
- Watch for consistent improvement over rounds
- Plateau indicates model convergence

### Loss
- Lower values indicate better performance
- Watch for consistent decrease over rounds
- Sudden increases may indicate overfitting

### Error Rates
- Lower values indicate better performance
- False Positive Rate (FPR): Proportion of negative examples incorrectly classified as positive
- False Negative Rate (FNR): Proportion of positive examples incorrectly classified as negative

## Analyzing Your Model's Performance

1. **Convergence**: Check if the loss curve flattens out, indicating model convergence.

2. **Precision-Recall Trade-off**: Higher precision often comes at the cost of lower recall and vice versa. The F1-score helps balance this trade-off.

3. **Error Analysis**: Compare FPR and FNR to understand if your model is biased toward predicting more positive or negative classes.

4. **Overall Improvement**: The summary statistics show how much your model improved from the first to the last round.

## Troubleshooting

If you encounter issues running the scripts:

1. Ensure your `results.csv` file exists and contains the expected columns
2. Try installing dependencies manually:
   ```
   pip install pandas matplotlib seaborn numpy
   ```
3. Check that you have write permissions in the `plots` directory 