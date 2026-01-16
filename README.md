# insurance-loss-prediction
Predicts insurance claim risk, loss cost (LC), and historically adjusted loss cost (HALC) using machine learning. Addresses zero-inflated, imbalanced data with feature engineering, ensembling, SMOTE, and SHAP to support underwriting, pricing, and risk management.

---

## Project Overview
This project develops machine learning models to support **insurance pricing, underwriting, and risk management** by predicting:

- **Claim Status (CS)** — whether a policyholder files a claim  
- **Loss Cost per Exposure Unit (LC)**  
- **Historically Adjusted Loss Cost (HALC)**  

The project addresses real-world insurance challenges such as **zero-inflated loss data, heavy right-skew, and severe class imbalance**, while balancing predictive performance with business interpretability.

---

## Business Motivation
Inaccurate pricing leads to **adverse selection**, where insurers lose low-risk customers and retain high-risk ones. Accurate prediction enables:

- Fair and competitive premium pricing  
- Early identification of high-risk policyholders  
- Improved portfolio profitability and risk segmentation  

---

## Data Description
- **Training set:** 37,451 auto insurance policies  
- **Test set:** 15,787 policies  
- **Features:** Policy tenure, payment behavior, cancellations, vehicle attributes, geographic indicators  

### Target Definitions
- **Loss Cost (LC)** = Total Claim Cost / Number of Claims  
- **Historically Adjusted Loss Cost (HALC)** = LC × Historical Adjustment Factor  
- **Claim Status (CS)** = 1 if a claim occurs, 0 otherwise  

The dataset exhibits **heavy zero-inflation, extreme right-skew, and strong class imbalance**.

---

## Feature Engineering
- Converted raw dates into:
  - Driver age  
  - License age  
  - Vehicle age  
  - Policy duration  
- One-hot encoded vehicle type, fuel source, and purchase channel  
- Capped extreme values to improve stability:
  - LC ≤ $20,000  
  - HALC ≤ $30,000  
- Applied **LASSO (L1 regularization)** for variable selection  

Skewed distributions were preserved to reflect real insurance risk dynamics rather than treated as noise.

---

## Modeling Approach

### Task 1: Loss Prediction (LC & HALC)
**Models evaluated:**
- Tweedie GLM  
- Neural Networks  
- LightGBM  
- XGBoost  

**Final Model: Weighted Ensemble**
- 95% Tweedie GLM  
- 5% XGBoost  

**Performance (RMSE):**
- LC: 441.70  
- HALC: 824.42  

This ensemble combines **actuarial interpretability** with **non-linear modeling power**.

---

### Task 2: Claim Status Prediction (CS)
**Key Challenge:** Severe class imbalance (~11% positive class)

**Solutions:**
- SMOTE applied to training data only  
- Business-aligned metric prioritization  

**Models evaluated:**
- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- XGBoost  

**Final Model: Tuned XGBoost Classifier**

**Performance:**
- Recall: 0.77  
- ROC-AUC: 0.72  

Recall was prioritized to minimize false negatives and reduce underpricing risk.

---

## Model Interpretability
SHAP was used to ensure **transparent and explainable predictions**.

Key drivers included:
- Policy duration  
- Net premium  
- Cancellation history  
- Payment frequency  
- Vehicle power, weight, and cylinder capacity  

These features align closely with real-world underwriting logic.

---

## Key Insights & Innovations
- Balanced accuracy and interpretability based on business use case  
- Combined actuarial rigor with ML flexibility via ensembling  
- Addressed zero-inflation and class imbalance without distorting business meaning  
- Proposed risk-type segmentation as a future enhancement  

---
