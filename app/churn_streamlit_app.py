import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px


st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

@st.cache_resource
def load_pipeline():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "pipeline.pkl")
    return joblib.load(model_path)

pipeline = load_pipeline()

st.title("Customer Churn Prediction App")

st.markdown("""
### Business Problem (ASK)
Customer churn directly impacts revenue and growth.  
This application predicts whether a customer is likely to churn, enabling **proactive retention strategies**.

---

### Model Overview
- **Dataset:** Telco Customer Churn
- **Target Variable:** Churn (Yes / No)
- **Model:** Logistic Regression
- **Pipeline:** Preprocessing + Encoding + Scaling + Model
- **Output:** Churn prediction and probability
""")

# Sidebar — Manual Input
st.sidebar.header("Enter Customer Details")

def user_input_features():
    data = {
        "gender": st.sidebar.selectbox("Gender", ["Male", "Female"]),
        "SeniorCitizen": st.sidebar.selectbox("Senior Citizen", [0, 1]),
        "Partner": st.sidebar.selectbox("Partner", ["Yes", "No"]),
        "Dependents": st.sidebar.selectbox("Dependents", ["Yes", "No"]),
        "tenure": st.sidebar.number_input("Tenure (months)", 0, 72, 12),
        "PhoneService": st.sidebar.selectbox("Phone Service", ["Yes", "No"]),
        "MultipleLines": st.sidebar.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        ),
        "InternetService": st.sidebar.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        ),
        "OnlineSecurity": st.sidebar.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        ),
        "OnlineBackup": st.sidebar.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        ),
        "DeviceProtection": st.sidebar.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        ),
        "TechSupport": st.sidebar.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        ),
        "StreamingTV": st.sidebar.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        ),
        "StreamingMovies": st.sidebar.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        ),
        "Contract": st.sidebar.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        ),
        "PaperlessBilling": st.sidebar.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        ),
        "PaymentMethod": st.sidebar.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        ),
        "MonthlyCharges": st.sidebar.number_input(
            "Monthly Charges", 0.0, 200.0, 70.0
        ),
        "TotalCharges": st.sidebar.number_input(
            "Total Charges", 0.0, 10000.0, 800.0
        )
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# CSV Upload Option
st.subheader("📂 Or Upload Customer Data (CSV)")

uploaded_file = st.file_uploader(
    "Upload a CSV file with the same feature columns",
    type=["csv"]
)

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.dataframe(input_df.head())

# Prediction
st.subheader("Prediction Results")

prediction = pipeline.predict(input_df)
prediction_proba = pipeline.predict_proba(input_df)

results_df = input_df.copy()
results_df["Churn Prediction"] = ["Yes" if p == 1 else "No" for p in prediction]
results_df["Churn Probability"] = prediction_proba[:, 1].round(3)

st.dataframe(results_df)

# Model Interpretability
st.subheader("📈 What Influences Churn?")

try:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    coefficients = model.coef_[0]
    feature_names = preprocessor.get_feature_names_out()

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients
    })

    # Top drivers
    top_positive = importance_df.sort_values(
        "Coefficient", ascending=False
    ).head(10)

    top_negative = importance_df.sort_values(
        "Coefficient"
    ).head(10)

    col1, col2 = st.columns(2)

    with col1:
        fig_pos = px.bar(
            top_positive,
            x="Coefficient",
            y="Feature",
            orientation="h",
            title="Top Features Increasing Churn"
        )
        st.plotly_chart(fig_pos, use_container_width=True)

    with col2:
        fig_neg = px.bar(
            top_negative,
            x="Coefficient",
            y="Feature",
            orientation="h",
            title="Top Features Reducing Churn"
        )
        st.plotly_chart(fig_neg, use_container_width=True)

except Exception as e:
    st.warning(
        "Feature importance unavailable for uploaded data. "
        "Ensure all columns match training data."
    )


st.markdown("""
### Business Actions (ACT)

- Customers with **high churn probability** should be targeted with retention offers.
- Month-to-month contracts are strong churn drivers → promote long-term plans.
- High charges without support services indicate dissatisfaction risk.
- Use predictions to **prioritize customer outreach** efficiently.

---

*This application demonstrates an end-to-end data science workflow:
ASK → PREPARE → PROCESS → MODEL → SHARE → ACT*
""")

