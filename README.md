# Customer Churn Prediction & Retention Analytics

End-to-end machine learning project that predicts which telecom customers are likely to churn, explains **why** they leave, and turns the findings into retention recommendations.

**Workflow:** Data → Statistics → EDA → Preprocessing → Feature Engineering → Modeling → Evaluation → Insights → Recommendations

## Business problem
Churn removes recurring revenue, and replacing a customer costs more than keeping one. Beyond predicting churn, this project identifies the drivers of churn, the customer segments that need attention, and practical actions to reduce it.

## Dataset
[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) – 7,043 customers, 21 columns (demographics, services, contract, billing, tenure, charges). Target: `Churn` (26.5% churned).
The CSV in `data/` is the unmodified public dataset (fetched from IBM's public GitHub mirror, which matches the Kaggle file: 7,043 rows, 21 columns). All cleaning is done in the notebook.

## Repository structure
```
├── Customer_Churn_Prediction.ipynb   # complete workflow, executed with outputs
├── data/Telco-Customer-Churn.csv     # raw dataset
├── outputs/churn_predictions.csv     # churn risk score + risk tier for every customer
├── images/                           # figures produced by the notebook
├── report/Project_Report.docx        # project report (PDF copy alongside)
├── requirements.txt
└── README.md
```

## How to run
```bash
git clone <your-repo-url>
cd customer-churn-prediction
pip install -r requirements.txt
jupyter notebook Customer_Churn_Prediction.ipynb     # then Run All
```
Or open the notebook in Google Colab; if `data/Telco-Customer-Churn.csv` is not found it downloads a public copy automatically. Runtime is a few minutes (grid searches). Random seed: 42.

## Approach in brief
| Stage | What was done |
|---|---|
| Data understanding | Shape, dtypes, duplicates (none), hidden blanks in `TotalCharges` (11 rows, all tenure = 0 → set to 0), inconsistent labels |
| Statistics | Mean, median, mode, min/max, variance, std, quartiles, IQR, skewness, outlier check (no outliers), explained in business terms |
| EDA | Univariate, bivariate, multivariate: histograms, box plots, count/bar plots, scatter, heatmaps |
| Preprocessing | Cleaning, one-hot encoding, scaling inside pipelines (no leakage), stratified 80/20 split |
| Feature engineering | 9 features (tenure group, add-on counts, support bundle flag, auto-pay, price-change proxy, …) with an ablation test |
| Models | Logistic Regression, Decision Tree, Random Forest, AdaBoost, KNN – baseline and tuned (GridSearchCV, F1) |
| Evaluation | Accuracy, precision, recall, F1, ROC-AUC, confusion matrices, CV stability, bootstrap CIs, calibration, threshold tuning |
| Business | Risk tiers, high-risk profile, cost-benefit simulation, recommendations |

## Key results
**Churn drivers** (churn rate by segment): month-to-month contract 43% (vs 3% two-year) · fibre optic 42% (DSL 19%) · electronic check 45% (automatic payment 15–17%) · no OnlineSecurity/TechSupport ~42% (with: ~15%) · first 12 months of tenure 47%.

**Final model:** tuned Logistic Regression (C = 0.03) with a recall-oriented threshold of 0.32 (chosen on out-of-fold training predictions only).

| Test set (1,409 customers) | Threshold 0.50 | Threshold 0.32 (final) |
|---|---|---|
| Recall | 0.786 | **0.925** |
| Precision | 0.514 | 0.439 |
| F1 | 0.622 | 0.595 |
| Accuracy | 0.746 | 0.666 |
| ROC-AUC | 0.840 | 0.840 |

Train and test scores are nearly identical (no overfitting). Tuned Random Forest scored slightly higher F1 (0.633) but overfits more (train–test gap 0.13) and is harder to explain, so Logistic Regression was chosen.
The High/Critical risk tiers are 40% of customers but contain ~79% of all churners.

## Honest notes and limitations
- Engineered features did **not** measurably improve accuracy (changes within CV noise); the raw features already carry most of the signal. They were kept for interpretability.
- All models plateau around ROC-AUC 0.81–0.84: the dataset lacks usage, complaint, network-quality and competitor-price data.
- Class weighting makes the score **over-state** absolute probabilities (mean 0.42 vs 26.5% actual). Use `churn_risk_score` to **rank** customers, not as a literal probability.
- At the final threshold, 56% of flagged customers would not have churned – retention offers should be cheap, and the threshold should be re-tuned to the real offer cost.
- Cost-benefit numbers use stated assumptions ($50 offer, 30% success, 12 months of value). Findings are associations, not causal effects; test offers with A/B experiments.

## Output file: `outputs/churn_predictions.csv`
One row per customer, sorted by risk: `customerID, Contract, InternetService, PaymentMethod, tenure, MonthlyCharges, churn_risk_score, predicted_churn, risk_tier, actual_churn`. Scores are out-of-fold (each customer scored by a model that did not train on them).

## Tech stack
Python · NumPy · Pandas · Matplotlib · Seaborn · Scikit-learn · Jupyter / Google Colab
