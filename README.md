# Mini Research Project: Employee Attrition Risk Prediction

This project is designed as a **mini research-style ML study**:
- A clear problem statement
- Reproducible dataset creation
- Model comparison
- Quantitative evaluation
- Basic visual analysis

## 1) Locked Idea

**Research question:**  
Can we predict employee attrition risk using a blend of behavioral and demographic features, and which baseline model performs better?

**Task type:** Binary classification (`attrition`: 0/1)

## 2) Dataset

The dataset is **synthetically generated** using `sklearn.make_classification` and post-processed into realistic HR-style columns:
- Numeric: `age`, `tenure_years`, `monthly_hours`, `overtime_hours`, `salary_usd`, `engagement_score`, `distance_km`, `projects_per_quarter`
- Categorical: `department`, `work_mode`, `education_level`, `satisfaction_band`
- Target: `attrition`

Generated file:
- `data/processed/employee_attrition.csv`

Why synthetic?
- Fully reproducible
- No privacy concerns
- Still realistic enough for methodology demonstration

## 3) Full Pipeline

### Step A - Data Creation
Script: `src/data/make_dataset.py`
- Generates 5,000 records
- Writes clean CSV to `data/processed/`

### Step B - Model Training + Evaluation
Script: `src/train.py`
- Train/test split with stratification
- Preprocessing via `ColumnTransformer`
  - Numeric: median imputation + scaling
  - Categorical: mode imputation + one-hot encoding
- Model comparison:
  - Logistic Regression
  - Random Forest
- Metrics:
  - Accuracy, Precision, Recall, F1, ROC-AUC
- Saves:
  - `artifacts/metrics.json`
  - `artifacts/test_predictions.csv`
  - `artifacts/models/best_model.joblib`

### Step C - Analysis Outputs
Script: `src/analyze.py`
- Creates figures:
  - `artifacts/figures/class_distribution.png`
  - `artifacts/figures/overtime_vs_attrition.png`
- Saves model comparison table:
  - `artifacts/model_comparison.csv`

## 4) Folder Structure

```text
ML Project/
  data/
    processed/
  artifacts/
    figures/
    models/
  src/
    data/make_dataset.py
    train.py
    analyze.py
  requirements.txt
  README.md
```

## 5) Quick Run

Use the commands from the chat response to:
1. Create a virtual environment
2. Install dependencies
3. Run data generation -> training -> analysis
4. Initialize git, commit, create GitHub repo, and push
