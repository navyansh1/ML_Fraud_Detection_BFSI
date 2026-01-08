# Credit Card Fraud Detection - BFSI Domain
📘 [View Notebook (Fast Web View)](https://nbviewer.org/github/navyansh1/ML_Fraud_Detection_BFSI/blob/main/BFSI%20transactionFraud%20analysis.ipynb)


[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost-green.svg)](https://xgboost.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

## Project Overview 📋

This project demonstrates **end-to-end fraud detection** in the Banking, Financial Services, and Insurance (BFSI) domain. Using machine learning, we identify fraudulent credit card transactions from a highly imbalanced dataset, achieving strong performance with practical business applications.

**Key Highlights:**

- ✅ Handles 99.6% class imbalance using advanced techniques
- ✅ Achieves 93%+ recall (catching most fraud cases)
- ✅ Provides business-interpretable feature importance
- ✅ Real-world applicable model with deployment considerations

---

## Problem Statement 🎯

Credit card fraud causes billions in losses annually. Traditional rule-based systems miss sophisticated fraud patterns. This project builds a machine learning model to:

1. **Detect fraudulent transactions** with high accuracy
2. **Minimize false positives** to avoid customer inconvenience
3. **Identify key fraud indicators** for business insights
4. **Handle extreme class imbalance** (fraud is rare)

---

## Dataset 📊

- **Source**: Simulated credit card transaction data
- **Size**: ~555,000 transactions
- **Class Distribution**: 99.6% legitimate, 0.4% fraud
- **Features**: Transaction amount, merchant category, location, time, customer demographics

**Key Features:**

- `amt`: Transaction amount
- `category`: Merchant category (e.g., shopping, gas, dining)
- `trans_date_trans_time`: Transaction timestamp
- `lat`, `long`: Customer location
- `merch_lat`, `merch_long`: Merchant location
- `is_fraud`: Target variable (0 = legit, 1 = fraud)

---

## Technical Approach 🛠️

### 1. **Data Preprocessing**

- Convert data types (dates, categorical identifiers)
- Handle missing values (none found in this dataset)
- Remove PII (personally identifiable information)

### 2. **Feature Engineering**

Created meaningful features from raw data:

- **Time features**: Hour, day, weekday from timestamp
- **Customer age**: Calculated from date of birth
- **Distance**: Geodesic distance between customer and merchant
- **Online indicator**: Flag for "card-not-present" transactions
- **Cyclical encoding**: Sin/cos transformation for time features
- **Frequency encoding**: For high-cardinality features (merchant)

### 3. **Exploratory Data Analysis**

- Analyzed class imbalance severity
- Identified fraud patterns by transaction amount, category, time
- Visualized feature distributions and correlations
- Detected and handled outliers

### 4. **Modeling**

**Primary Model: XGBoost with Class Weighting**

- **Why XGBoost?** Handles imbalanced data, fast, interpretable
- **Imbalance handling**: `scale_pos_weight` parameter (not SMOTE)
- **Hyperparameters**: 300 trees, max depth 6, learning rate 0.1

**Alternative approaches considered:**

- SMOTE + Logistic Regression
- SMOTE + Random Forest
- Decision Tree (for interpretability)

### 5. **Evaluation**

**Metrics focus:**

- **Recall (93%+)**: Catching most fraud cases (minimizes financial loss)
- **Precision**: Reducing false alarms (customer experience)
- **ROC-AUC (0.997)**: Overall separation ability
- **Business impact**: Actual fraud caught vs. missed

---

## Results 📈

### Model Performance (Test Set)

| Metric                    | Value | Business Meaning                       |
| ------------------------- | ----- | -------------------------------------- |
| **Recall**          | 93%+  | Catching 93% of all frauds             |
| **Precision**       | 51%   | Half of fraud alerts are genuine       |
| **ROC-AUC**         | 0.997 | Excellent separation                   |
| **False Negatives** | ~32   | Missed frauds (opportunity to improve) |
| **False Positives** | ~375  | False alarms (customer friction)       |

### Key Fraud Indicators

**Top 5 Features (by importance):**

1. **Transaction Amount**: Higher amounts = higher risk
2. **Online Indicator**: Card-not-present transactions riskier
3. **Merchant Frequency**: New/rare merchants increase risk
4. **Time Features**: Fraud patterns vary by hour/day
5. **Distance**: Unusual location activity

---

## Business Impact 💼

### Financial Savings

- **Frauds Caught**: ~397 out of 429 (93%)
- **Frauds Missed**: ~32 (7% - opportunity for improvement)
- **Potential Savings**: Millions in prevented fraudulent transactions

### Customer Experience

- **False Alarms**: ~375 (0.3% of legitimate transactions)
- **Trade-off**: High fraud detection vs. minimal customer inconvenience

### Actionable Insights

1. **Online transactions** need extra scrutiny (higher fraud rate)
2. **High-value transactions** should trigger additional verification
3. **Unusual merchant activity** is a strong fraud signal
4. **Time-based patterns** can optimize fraud detection systems

---

## How to Use This Project ?

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost geopy
```

### Running the Notebook

1. **Clone/Download** this repository
2. **Place your data**: Ensure `fraudTest.csv` is in the same directory
3. **Open Jupyter**: `jupyter notebook "BFSI transactionFraud analysis.ipynb"`
4. **Run all cells**: The notebook is organized sequentially

### File Structure

```
BFSI/
├── BFSI transactionFraud analysis.ipynb  # Main analysis notebook
├── README.md                              # This file
└── fraudTest.csv                          # Dataset (not included)
```

---

## Key Learnings 📚

### Technical Learnings

1. **Class imbalance** requires special handling - use class weighting or SMOTE
2. **Feature engineering** dramatically improves model performance
3. **Cyclical encoding** (sin/cos) preserves time relationships
4. **Tree-based models** (XGBoost) handle raw features well (no scaling needed)

### Domain Learnings

1. **Recall > Precision** in fraud detection - missing fraud is costly
2. **False positives** affect customer satisfaction - balance is key
3. **Interpretability** builds trust - feature importance matters
4. **Real-time scoring** requires fast, scalable models

### Process Learnings

1. **Train/Validation/Test split** prevents overfitting
2. **Business metrics** matter more than technical metrics
3. **Visualization** communicates insights effectively
4. **Documentation** (like this README) makes projects shareable

---

## 🔮 Future Improvements

1. **Hyperparameter Tuning**: GridSearchCV for optimal parameters
2. **Ensemble Methods**: Combine multiple models
3. **Real-time Deployment**: Flask API or cloud deployment
4. **Continuous Learning**: Retrain with new fraud patterns
5. **Explainability**: SHAP values for individual predictions
6. **Cost-sensitive Learning**: Assign different costs to errors

