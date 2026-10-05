# Digital Dependence Risk Prediction

This project explores digital dependence and predicts a continuous
`digital_dependence_score` from demographic, device-use, wellbeing, and
behavioral features. It also investigates the binary `high_risk_flag` target
and uses explainable AI techniques to interpret the trained models.

The main analysis is implemented in [`Ml.ipynb`](./Ml.ipynb), and the written
analysis is available in [`ML Report.pdf`](./ML%20Report.pdf).

## Project workflow

The notebook covers:

1. Loading the dataset from Google Drive.
2. Exploratory data analysis (EDA), including distributions, missing values,
   duplicates, data types, cardinality, and correlations.
3. Categorical encoding and class-imbalance analysis.
4. Synthetic Minority Over-sampling Technique (SMOTE) for the
   `high_risk_flag` classification data.
5. Outlier detection with IQR and z-score methods, followed by IQR clipping.
6. Feature engineering, including:
   - Mental health index
   - Sleep-stress ratio
   - Notification density
   - Age groups
   - Heavy-user indicator
7. An 80/20 train-test split.
8. Regression experiments with:
   - Random Forest
   - AdaBoost
   - Scikit-learn MLP
   - TensorFlow/Keras neural network
9. Model evaluation using mean squared error, R-squared, and cross-validation.
10. Model interpretation with SHAP, LIME, and partial dependence plots.

## Requirements

Use Python 3.9 or newer with Jupyter Notebook or Google Colab. The notebook
uses the following packages:

```text
pandas
numpy
scipy
scikit-learn
imbalanced-learn
matplotlib
seaborn
shap
lime
tensorflow
```

Install the dependencies with:

```bash
pip install pandas numpy scipy scikit-learn imbalanced-learn matplotlib seaborn shap lime tensorflow jupyter
```

## Running the notebook

1. Open [`Ml.ipynb`](./Ml.ipynb) in Jupyter Notebook, JupyterLab, or Google
   Colab.
2. Install the dependencies listed above.
3. Run the cells from top to bottom.
4. When prompted or when the dataset-loading cell is reached, make sure the
   dataset URL is accessible. The notebook currently loads the data from a
   Google Drive link embedded in the notebook.

The notebook contains model-training and visualization cells that may take
several minutes to run, especially SHAP, LIME, and neural-network sections.

## Data

The dataset is not stored in this repository. The notebook retrieves it from
Google Drive and expects the loaded dataframe to contain fields such as:

- `digital_dependence_score`
- `high_risk_flag`
- `age`
- `gender`
- `region`
- `income_level`
- `education_level`
- `daily_role`
- `device_type`
- device-use, sleep, stress, anxiety, depression, happiness, focus, and
  productivity measures

If the source link is unavailable, replace the data-loading cell with a local
CSV path or another authorized dataset source. Do not commit private or
sensitive data to the repository.

## Outputs

Running the notebook produces:

- EDA tables and plots
- Correlation and class-balance visualizations
- Outlier summaries and feature-engineering results
- Regression metrics and model-comparison tables
- Feature-importance plots
- SHAP global and local explanations
- LIME local explanations
- Partial dependence plots

## Reproducibility notes

The notebook uses `random_state=42` in the main train-test splits and model
experiments. Results can still vary with package versions, TensorFlow
hardware/runtime behavior, and changes to the upstream dataset.

## Repository contents

```text
.
├── Ml.ipynb       # End-to-end analysis and model experiments
├── ML Report.pdf  # Project report
└── README.md      # Project documentation
```

## License

No license has been specified yet. Add a license before distributing or
reusing this project publicly.
