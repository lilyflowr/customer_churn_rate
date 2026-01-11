# Customer Churn Prediction (Telco)
## End-to-End Applied Data Science Project

 

## Overview

Customer churn is a major revenue risk for subscription-based businesses.  
In this project, I built an **end-to-end predictive churn system** that identifies customers at risk of leaving and enables **real-time interaction with predictions** via a deployed Streamlit application.

This case study demonstrates **applied data science capability** — from problem framing and feature engineering to modeling, interpretability, and stakeholder-facing deployment.

 

## Business Problem (ASK)

### Key Questions
- How can we predict which customers are most likely to churn?
- Which features most strongly influence churn risk?
- How can the business act on churn predictions?

### Why This Matters
Retaining existing customers is significantly cheaper than acquiring new ones.  
A reliable churn prediction model enables **proactive retention strategies** rather than reactive loss management.

 

## Project Objectives

- Build a **binary classification model** to predict churn
- Identify **key drivers of churn** using interpretable modeling
- Translate predictions into **business-actionable insights**
- Deploy an **interactive application** for stakeholder use

 

## Dataset (PREPARE)

- **Dataset:** Telco Customer Churn  
- **Source:** IBM Sample Dataset (https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- **Observations:** ~7,000 customers  
- **Target Variable:** `Churn` (Yes / No)

### Feature Categories
- **Demographics:** gender, senior citizen, dependents  
- **Account details:** tenure, contract type, payment method  
- **Services:** internet, phone, streaming, support  
- **Billing:** monthly charges, total charges  

### Data Considerations
- Mixed numerical and categorical features
- Missing values introduced during numeric coercion
- Class imbalance present (churn vs non-churn)

 

## Data Processing & Feature Engineering (PROCESS)

All preprocessing was handled using a **scikit-learn Pipeline** to ensure:

- Reproducibility
- No data leakage
- Consistency between training and deployment

### Key Steps
- Type coercion for numeric billing fields
- Median imputation for numerical features
- Most-frequent imputation for categorical features
- One-Hot Encoding for categorical variables
- Standard scaling for numerical variables

✔ No manual preprocessing outside the pipeline  
✔ Same transformations used in training and inference

 

## Modeling Approach (MODEL)

### Model Used
- **Logistic Regression**

### Why Logistic Regression?
- Strong baseline performance
- High interpretability
- Clear business explainability through coefficients

The goal was **decision support**, not just accuracy.

### Evaluation Focus
- Precision & Recall (churn identification)
- ROC-AUC
- Confusion Matrix
- Feature coefficient analysis

 

##  Key Insights (ANALYZE)

### Features Increasing Churn Risk
- Month-to-month contracts
- Electronic check payments
- Fiber optic internet without support services
- Short customer tenure

### Features Reducing Churn Risk
- Long-term contracts (1–2 years)
- Tech support and online security
- Automatic payment methods

These insights directly inform **targeted retention strategies**.

 

## Interactive Deployment (SHARE)

To move beyond static dashboards, the model was deployed as an **interactive Streamlit application**.

### App Capabilities
- Manual customer data entry
- CSV upload for batch predictions
- Churn probability scoring
- Real-time prediction output
- Feature influence visualization
- Business-friendly interface


 

## Running the App Locally

streamlit run churn_streamlit_app.

## Business Impact (ACT)

### How the Model Can Be Used

- Identify high-risk customers early  
- Target retention campaigns effectively  
- Optimize customer support allocation  
- Reduce revenue loss due to churn  

**Action:**  
Customers with **churn probability > 70%** can be targeted with contract upgrades or support bundles.

 

## Future Improvements

- Cost-sensitive modeling (optimize for retention ROI)  
- Threshold tuning based on business constraints  
- Tree-based models (XGBoost, LightGBM)  
- SHAP-based interpretability  
- Cloud deployment (Streamlit Cloud)

 

## Skills Demonstrated

- Applied Data Science  
- Predictive Modeling  
- Feature Engineering  
- Model Interpretability  
- Pipeline Design  
- Deployment & Stakeholder Interactivity  
- Business-driven analytics  

 

## 🔗 Links

- **Medium Case Study:** (https://ibinaboadiela.medium.com/) 
- **Portfolio:** (https://ibinaboadiela.vercel.app/)  
- **LinkedIn:** (https://www.linkedin.com/in/ibinaboadiela/)
