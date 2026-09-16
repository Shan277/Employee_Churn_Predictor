# Employee Churn Predictor

Predict whether an employee is likely to leave the company using HR analytics data, with a Streamlit app for interactive predictions.

## 🛠️ Tools & Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Pandas](https://img.shields.io/badge/Pandas-DataFrame-150458)
![NumPy](https://img.shields.io/badge/NumPy-Array-013243)
![scikit--learn](https://img.shields.io/badge/scikit--learn-Pipeline%20%7C%20RandomForest-F7931E)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-EB0000)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plotting-11557C)
![Seaborn](https://img.shields.io/badge/Seaborn-EDA-4C72B0)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B)
![Joblib](https://img.shields.io/badge/Joblib-Model%20Serialization-8A2BE2)

## 📌 Overview

This project analyzes HR data to understand why employees leave a company and builds a machine learning model to predict future turnover. It includes:

- Exploratory Data Analysis (EDA) to surface key churn drivers
- A preprocessing + modeling pipeline (Random Forest vs. XGBoost)
- A deployed Streamlit app for real-time, single-employee predictions

## 📊 Dataset

The dataset (`HR_comma_sep.csv`) contains ~15,000 employee records with the following fields:

| Feature | Description |
|---|---|
| `satisfaction_level` | Employee satisfaction score (0–1) |
| `last_evaluation` | Score from last performance review (0–1) |
| `number_project` | Number of projects assigned |
| `average_montly_hours` | Average monthly working hours |
| `time_spend_company` | Years spent at the company |
| `Work_accident` | Whether the employee had a workplace accident (0/1) |
| `promotion_last_5years` | Promoted in the last 5 years (0/1) |
| `Department` | Department the employee works in |
| `salary` | Salary level (low / medium / high) |
| `left` | Target — whether the employee left (1) or stayed (0) |

## 🔍 Exploratory Data Analysis

Key steps performed in the notebook:
- Checked shape, data types, and summary statistics
- Removed duplicate records
- Visualized feature distributions
- Analyzed churn against satisfaction, working hours, salary level, project count, and promotions
- Correlation heatmap across numerical features

## ⚙️ Preprocessing & Modeling Pipeline

Built using `scikit-learn`'s `ColumnTransformer` and `Pipeline`:

- **Numerical features** → scaled with `StandardScaler`
- **`Department`** → one-hot encoded
- **`salary`** → ordinal encoded (`low` < `medium` < `high`)

Two models were trained and compared:

| Model | Notes |
|---|---|
| Random Forest Classifier | `class_weight="balanced"` to handle class imbalance |
| XGBoost Classifier | Gradient-boosted trees |

Evaluated using **Accuracy, Precision, Recall, and F1 Score**, with the final model serialized using `joblib` (`model.pkl`).

## 🖥️ Streamlit App

`app.py` loads the trained model and provides a simple UI to input an employee's details (satisfaction level, working hours, tenure, department, salary, etc.) and returns a prediction of whether that employee is likely to stay or leave.

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/employee-churn-predictor.git
cd employee-churn-predictor
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit joblib
```

### 3. Train the model (optional — a pretrained `model.pkl` can be used instead)
Run through `employee-churn-detector.ipynb` to reproduce the EDA and train the model.

### 4. Run the app
```bash
streamlit run app.py
```

## 📂 Project Structure

```
├── employee-churn-detector.ipynb   # EDA, preprocessing, model training & evaluation
├── app.py                          # Streamlit app for predictions
├── model.pkl                       # Serialized trained model
└── README.md
```

## 📈 Possible Improvements

- Hyperparameter tuning (GridSearchCV / Optuna) for both models
- Feature importance visualization for interpretability
- Cross-validation for more robust performance estimates
- Deploy the app (Streamlit Community Cloud / Docker)

## 📝 License

This project is open-sourced for educational purposes.
