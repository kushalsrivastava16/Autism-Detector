# 🧩 Autism Prediction Using Machine Learning

A machine learning project to predict Autism Spectrum Disorder (ASD) using behavioral and demographic data. Three classification algorithms are compared — Decision Tree, Random Forest, and XGBoost — with hyperparameter tuning to find the best-performing model.

---

## 📂 Dataset

The dataset is sourced from Kaggle: [Autism Diagnosis Dataset](https://www.kaggle.com/datasets/faizunnabi/autismdiagnosis).

It contains behavioral screening answers (A1–A10 scores), demographic information (age, gender, ethnicity, country of residence), and a binary target column `Class/ASD` indicating whether a person has autism.

---

## 🔧 Project Pipeline

### 1. Data Cleaning
- Removed uninformative columns (`ID`, `age_desc`)
- Standardized country names (e.g., Hong Kong → China)
- Handled missing/invalid values in `ethnicity` and `relation` columns

### 2. Exploratory Data Analysis (EDA)
- Distribution plots for numerical features (`age`, `result`)
- Box plots for outlier detection
- Count plots for all categorical features
- Correlation heatmap for bivariate analysis

### 3. Data Preprocessing
- **Label Encoding** for categorical columns (encoders saved as `encoders.pkl`)
- **Outlier handling** using IQR-based median replacement
- **SMOTE** (Synthetic Minority Oversampling Technique) to address class imbalance
- 80/20 Train-Test Split

### 4. Model Training & Selection
Three models trained with 5-fold cross-validation:
| Model | Description |
|---|---|
| Decision Tree | Baseline tree-based classifier |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient boosted trees |

Hyperparameter tuning done using `RandomizedSearchCV` across all three models. The best model is automatically selected by cross-validation accuracy.

### 5. Model Evaluation
- Accuracy Score
- Confusion Matrix (visualized as a heatmap)
- Classification Report (Precision, Recall, F1-Score)

The best model is saved as `best_model.pkl` for later use.

---

## 🛠️ Tech Stack

- **Python**
- **Pandas**, **NumPy** — data manipulation
- **Matplotlib**, **Seaborn** — visualization
- **Scikit-learn** — preprocessing, modeling, evaluation
- **XGBoost** — gradient boosting classifier
- **imbalanced-learn (SMOTE)** — handling class imbalance
- **Pickle** — model and encoder serialization

---

## 🚀 Getting Started

```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn

# Run the notebook
jupyter notebook autism-prediction-using-different-algorithms.ipynb
```

Download the dataset from Kaggle and place it at:
```
/kaggle/input/autismdiagnosis/Autism_Prediction/train.csv
/kaggle/input/autismdiagnosis/Autism_Prediction/test.csv
```

---

## 📁 Output Files

| File | Description |
|---|---|
| `best_model.pkl` | Best trained model (selected automatically) |
| `encoders.pkl` | Label encoders for categorical columns |
