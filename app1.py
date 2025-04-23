import streamlit as st
import pandas as pd
import joblib

# Paths to your dataset and trained model
DATA_PATH = "small_dataset.csv"
MODEL_PATH = "rf_fraud_model.joblib"

# Load data with Streamlit caching
@st.cache_data
def load_data(path=DATA_PATH):
    return pd.read_csv(path)

# Load model with Streamlit caching
@st.cache_resource
def load_model(path=MODEL_PATH):
    return joblib.load(path)

# Make prediction based on cc_num and trans_num
def predict_by_keys(cc_num: int, trans_num: str, df: pd.DataFrame, model_bundle):
    subset = df[(df["cc_num"] == cc_num) & (df["trans_num"] == trans_num)]
    if subset.empty:
        return None, None
    X_query = subset.drop(columns=["is_fraud"])
    model = model_bundle["model"]
    pred = model.predict(X_query)[0]
    proba = model.predict_proba(X_query)[0][1]
    return int(pred), proba

# ----------------------------
# Streamlit App UI Starts Here
# ----------------------------
st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("Credit Card Fraud Detection")

st.write("Enter a credit card number and transaction number to check if it's fraudulent.")

cc_num = st.text_input("Credit Card Number")
trans_num = st.text_input("Transaction Number")

if st.button("Predict"):
    if not cc_num or not trans_num:
        st.warning("Please enter both Credit Card Number and Transaction Number.")
    else:
        try:
            cc_num = int(cc_num)
            df = load_data()
            model_bundle = load_model()
            pred, proba = predict_by_keys(cc_num, trans_num, df, model_bundle)
            if pred is None:
                st.error("No matching record found for the provided details.")
            else:
                label = "FRAUD" if pred == 1 else "LEGIT"
                st.success(f"Prediction: *{label}*")
                st.info(f"Fraud Probability: *{proba:.4f}*")
        except ValueError:
            st.error("Credit Card Number must be numeric.")
