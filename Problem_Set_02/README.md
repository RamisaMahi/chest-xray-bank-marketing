# Problem Set 02: Bank Term Deposit Subscription Prediction

## 1. Objective
Build a interpretable **Logistic Regression** classifier to predict whether a customer will subscribe to a bank term deposit (`y = yes/no`) based on 17 demographic, financial, and campaign behavioral features.

## 2. Preprocessing & Engineering Methodology
- **Encoding**: Categorical features (e.g., `job`, `marital`, `education`, `contact`) are transformed using **One-Hot Encoding** with `drop='first'` to prevent multicollinearity (dummy variable trap).
- **Scaling**: Continuous numerical variables (e.g., `age`, `balance`, `duration`, `pdays`) are normalized using **StandardScaler** to ensure uniform gradient descent scaling.
- **Class Imbalance Handling**: Since term deposit subscriptions constitute a minority class (~11-12%), the `class_weight='balanced'` parameter is utilized in the Logistic Regression estimator to penalize misclassifying positive instances.

## 3. Findings & Insights
1. **Last Contact Duration (`duration`)**: Strongest positive predictor. Longer conversations directly correlate with higher conversion probability.
2. **Outcome of Previous Campaign (`poutcome_success`)**: Customers who accepted past offers exhibit significantly higher odds of subscribing again.
3. **Economic Indicators & Contacts**: Frequent contact during the current campaign (`campaign` count) without success yields negative coefficients, indicating campaign fatigue.

## 4. Model Evaluation
- **ROC-AUC Score**: ~0.88 - 0.90 (indicates strong class discrimination capability).
- Balanced precision and recall trade-offs achieved via weighted loss optimization.
