# Customer Churn Prediction (Telco)

 

## Overview

Customer churn is a persistent revenue risk for subscription-based businesses, especially in industries where switching costs are low and acquisition is expensive.

This project focuses on identifying customers at elevated risk of churn and translating that risk into prioritized, interpretable, and actionable retention decisions. The emphasis is on building a churn assessment that business teams can understand, trust, and use.

An interactive application was deployed to ensure the analysis is accessible beyond notebooks and static reports.


## Business Framing
Retention teams rarely struggle with data availability. The real challenge is deciding who to act on, when to intervene, and which levers are worth pulling.

This project addresses three core questions:
- Which customers show the strongest signals of churn risk?
- What patterns consistently explain that risk?
- How can churn probabilities support prioritization rather than generic outreach?

 
## Dataset Context

- **Dataset:** Telco Customer Churn  
- **Source:** IBM Sample Dataset (https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- **Population:** ~7,000 customers  
- **Target Variable:** `Churn` (Yes / No)

## Analytical Approach
All preprocessing and modeling steps were handled through a scikit-learn pipeline to ensure consistency between training and deployment. This eliminated manual transformations and reduced the risk of leakage or mismatch during inference.

A logistic regression model was selected intentionally. The goal was to produce probability-based risk scores that can be explained to non-technical stakeholders and used directly for prioritization.

### What Drives Churn Risk
- Customers on month-to-month contracts show materially higher churn risk than those on longer-term agreements.
- Short tenure is one of the strongest indicators of early attrition.
- Customers paying via electronic check and those without support-related add-on services exhibit higher likelihood of churn.
- Long-term contracts, automatic payments, and bundled support services are associated with lower churn risk
 

### Action
Instead of producing a binary churn label, the model outputs churn probabilities. This enables:

- Ranking customers by risk

- Differentiating retention strategies by risk tier

- Allocating retention resources where expected impact is highest

For example, customers with churn probabilities above a defined threshold can be flagged for proactive outreach, contract incentives, or service upgrades, while lower-risk segments require minimal intervention. 

## Deployment
To support practical use, the model was deployed as an interactive Streamlit application.

The app allows users to:

- Input individual customer profiles

- Upload customer datasets for batch scoring

- View churn probabilities alongside feature influence

- Explore how changes in contract or service attributes affect risk
 

## How This Can Be Extended
Future improvements could include:

- Cost-sensitive threshold tuning aligned with retention budget 
- ROI-based intervention simulation
- Tree-based models for comparison against the interpretable baseline
- Integration with live customer systems for automated scoring


## 🔗 Links
- **The App:** (https://customerchurnrateprediction.streamlit.app/) 
- **Medium Case Study:** (https://ibinaboadiela.medium.com/) 
- **Portfolio:** (https://ibinaboadiela.vercel.app/)  
- **LinkedIn:** (https://www.linkedin.com/in/ibinaboadiela/)
