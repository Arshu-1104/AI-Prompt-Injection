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

```powershell
cd "C:\Users\Admin\Desktop\ML Project"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt

py src/data/make_dataset.py
py src/train.py
py src/analyze.py
```

If `py` is unavailable in your setup, replace `py` with `python`.

## 6) Current Best Results

From `artifacts/metrics.json`:
- Best model: `random_forest`
- Accuracy: `0.901`
- Precision: `0.873`
- Recall: `0.759`
- F1: `0.812`
- ROC-AUC: `0.955`

## 7) Teacher Demo: How to Show Output

Use this sequence in front of your teacher:

1. Run the full pipeline
```powershell
py src/data/make_dataset.py
py src/train.py
py src/analyze.py
```

2. Show the key generated files
```powershell
Get-ChildItem .\data\processed\
Get-ChildItem .\artifacts\
Get-ChildItem .\artifacts\models\
Get-ChildItem .\artifacts\figures\
```

3. Show model metrics clearly
```powershell
Get-Content .\artifacts\metrics.json
```

4. Show concise research report
```powershell
Get-Content .\report.md
```

5. Explain in one line:
- "We compared Logistic Regression and Random Forest on a reproducible HR-style dataset; Random Forest performed best with F1 = 0.812 and ROC-AUC = 0.955."

## 8) Git + GitHub

Local repo commands:
```powershell
git add .
git commit -m "Add polished report and teacher-ready README demo workflow"
```

Push commands (after setting your correct repo URL):
```powershell
git branch -M main
git remote set-url origin https://github.com/<your-username>/employee-attrition-mini-research.git
git push -u origin main
```
