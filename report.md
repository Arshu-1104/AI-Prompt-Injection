# Mini Research Report

## Title
Employee Attrition Risk Prediction Using Baseline Machine Learning Models

## Abstract
This project investigates whether employee attrition can be predicted from synthetic HR-style features that combine work behavior, demographics, and job context. Two baseline classifiers (Logistic Regression and Random Forest) are compared under a consistent preprocessing pipeline. Random Forest achieves the best overall performance, with strong discrimination and balanced minority-class detection.

## Problem Statement
Organizations face productivity and retention costs due to employee attrition. The objective is to predict attrition risk (`0` = stay, `1` = leave) and identify a practical baseline model for early risk screening.

## Dataset
- Source: synthetic dataset generated with `sklearn.make_classification`, then mapped to realistic HR fields
- Rows: 5000
- Target distribution: attrition rate = 28.22%
- Feature groups:
  - Numeric: age, tenure, monthly hours, overtime, salary, engagement, commute distance, project load
  - Categorical: department, work mode, education level, satisfaction band

Synthetic data is used to ensure reproducibility and avoid privacy constraints while preserving realistic structure.

## Methodology
1. Generate and store cleaned tabular data.
2. Perform stratified train-test split (80/20).
3. Apply preprocessing:
   - Numeric: median imputation + standard scaling
   - Categorical: mode imputation + one-hot encoding
4. Train and compare:
   - Logistic Regression
   - Random Forest
5. Evaluate with Accuracy, Precision, Recall, F1-score, ROC-AUC.

## Results
### Model Comparison
- Logistic Regression: Accuracy 0.808, Precision 0.684, Recall 0.592, F1 0.635, ROC-AUC 0.849
- Random Forest: Accuracy 0.901, Precision 0.873, Recall 0.759, F1 0.812, ROC-AUC 0.955

### Best Model
Random Forest is selected by best F1-score.

Confusion Matrix (test set):
- True Negatives: 687
- False Positives: 31
- False Negatives: 68
- True Positives: 214

Interpretation: The model captures attrition cases well while keeping false alarms relatively low.

## Behavioral Insight
Average overtime differs strongly by class:
- Attrition = 1: 19.61 hours
- Attrition = 0: 8.64 hours

This supports overtime as an informative signal for attrition risk.

## Limitations
- Data is synthetic, so external validity is limited.
- No hyperparameter search beyond baseline settings.
- No fairness or calibration audit included.

## Future Work
- Add cross-validation and systematic hyperparameter tuning.
- Add calibration curves and threshold optimization.
- Add explainability (feature importance, SHAP).
- Evaluate fairness across department/work-mode groups.

## Reproducibility Artifacts
- `data/processed/employee_attrition.csv`
- `artifacts/metrics.json`
- `artifacts/model_comparison.csv`
- `artifacts/test_predictions.csv`
- `artifacts/models/best_model.joblib`
