# Re-analysis of "A novel cortical biomarker signature predicts individual pain sensitivity"

This repository contains code for the re-analysis of the preprint: **"A novel cortical biomarker signature predicts individual pain sensitivity"**. The analysis includes the preprocessing, merging, and modeling of data to evaluate predictive models for individual pain sensitivity class (low vs high) based on peak-alpha frequency and corticomotor excitability.

## Overview

The code performs the following tasks:
1. **Data Loading and Preprocessing**: Loads data from Excel files, calculates CME (Chronic Morphological Effect), and merges datasets on individual IDs.
2. **Model Training and Evaluation**: Trains several machine learning models (e.g., Logistic Regression, Random Forest, Gradient Boosting, SVC, MLPClassifier) using nested pipelines, imputing missing values and scaling features.
3. **Hyperparameter Tuning**: Optimizes models using grid search with cross-validation.
4. **Performance Evaluation**: Computes accuracy and AUC metrics for training and test datasets.
5. **Result Aggregation**: Aggregates results across multiple random seeds and saves them to CSV files.
6. **Visualization**: Generates bar plots for accuracy and AUC metrics for each model.

## Requirements

To run this code, you need the following Python libraries:
- numpy
- pandas
- scikit-learn
- seaborn
- matplotlib
- openpyxl (for reading Excel files)

Ensure you have Python 3.8 or higher installed.

### Installation
You can install the required libraries using:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:

```txt
numpy
pandas
scikit-learn
seaborn
matplotlib
openpyxl
```

## How to Run the Code

1. Clone this repository:

   ```bash
   git clone https://github.com/your_username/pain-sensitivity-reanalysis.git
   cd pain-sensitivity-reanalysis
   ```

2. Set the `basepath` variable to the directory containing your input data files. The expected files are:
   - `PAF_all.xlsx`: Contains peak-alpha frequency data.
   - `map_volume_all.xlsx`: Contains volume data for calculating CME.
   - `class_IDs_all.xlsx`: Contains class labels for individuals.

3. Run the Jupyter notebook or Python script. For example:

   ```bash
   jupyter notebook
   ```

4. Execute the cells in order, ensuring the data files are in the specified `basepath`.

5. Results will be saved in CSV format:
   - `results_all_runs.csv`: Contains detailed performance metrics for each random seed and model.
   - `summary_results.csv`: Contains aggregated accuracy and AUC metrics for training and test datasets.

6. Visualization outputs are saved as SVG files:
   - `Accuracy_by_Model.svg`
   - `AUC_by_Model.svg`

## Contact
For questions or issues, please contact [ole.goltermann@maxplanckschools.de].

