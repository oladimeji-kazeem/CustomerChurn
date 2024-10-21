import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained model
model_path = './models/churn_model.pkl'  # Adjust this path if needed
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Insert the logo at the top
st.image("image/logo.png", width=100)  # Replace "logo.png" with your logo file name and adjust the width if needed

# Function to make predictions for a single input
def predict_churn(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    prediction_proba = model.predict_proba(input_array)
    return prediction[0], prediction_proba[0][1]

# Function to process batch predictions
def batch_predict(df):
    # Apply the prediction logic to each row of the dataframe
    df['ChurnPrediction'], df['ChurnProbability'] = zip(*df.apply(lambda row: predict_churn([
        row['Age'], 
        1 if row['Gender'] == 'Male' else 0,  # Assuming Male = 1, Female = 0
        1 if row['MaritalStatus'] == 'Married' else 0,  # Example encoding
        0,  # PolicyType (map this appropriately)
        row['Premium'], 
        row['NumClaims'], 
        row['ClaimSatisfaction'], 
        0,  # PaymentHistory (map appropriately)
        0,  # Product (map appropriately)
        1 if row['RenewalStatus'] == 'Yes' else 0, 
        1 if row['CustomerCategory'] == 'Individual' else 0,
        0,  # Region (map appropriately)
        0,  # State (map appropriately)
        row['SomeOtherFeature']  # <-- This is the missing 14th feature, ensure it matches your training data
    ]), axis=1))
    
    return df

# Streamlit app
st.title("Customer Churn Prediction")

# Option to download the template CSV
st.write("#### Download Template")
template_data = {
    'Age': [],
    'Gender': [],
    'MaritalStatus': [],
    'PolicyType': [],
    'Premium': [],
    'NumClaims': [],
    'ClaimSatisfaction': [],
    'PaymentHistory': [],
    'Product': [],
    'RenewalStatus': [],
    'CustomerCategory': [],
    'Region': [],
    'State': [],
    'SomeOtherFeature': []  # Ensure this is added to the template
}
template_df = pd.DataFrame(template_data)

# Provide option to download the CSV template
st.download_button(
    label="Download Template (CSV)", 
    data=template_df.to_csv(index=False), 
    file_name='churn_prediction_template.csv', 
    mime='text/csv'
)

# File upload section for batch processing
st.write("#### Upload File for Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    # Read the uploaded file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Perform batch predictions
    results_df = batch_predict(df)

    # Show results
    st.write("Predictions:")
    st.dataframe(results_df)

    # Provide option to download the results
    st.download_button(
        label="Download Predictions (CSV)", 
        data=results_df.to_csv(index=False), 
        file_name='churn_predictions.csv', 
        mime='text/csv'
    )

# Allow single record prediction as well
st.write("#### Single Record Prediction")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
policy_type = st.selectbox("Policy Type", ["Health", "Life", "Auto", "Home", "Travel"])
premium = st.number_input("Premium (₦)", min_value=10000, max_value=500000, value=100000)
num_claims = st.number_input("Number of Claims", min_value=0, max_value=10, value=0)
claim_satisfaction = st.slider("Claim Satisfaction", min_value=1, max_value=5, value=3)
payment_history = st.selectbox("Payment History", ["On-time", "Late", "Missed"])
product = st.selectbox("Product", [
    'Motor 3rd Party', 'Marine Insurance', 'Machine Breakdown', 'All Risk Insurance', 
    'Aviation', 'Professional Indemnity', 'Burglary', 'Comprehensive Motor', 
    'Consequential Loss', 'Goods-In-Transit', 'Enhanced Motor 3rd Party', 'Oil & Gas', 
    'Contractors-All-Risk', 'Plant-All-Risk', 'Bonds Insurance', 'Personal Accident', 
    'Travel', 'Home', 'Fire & Special Perils', 'Fire & Burglary'
])
renewal_status = st.selectbox("Renewal Status", ["Yes", "No"])
customer_category = st.selectbox("Customer Category", ["Individual", "Corporate"])
region = st.selectbox("Region", ['NorthEast', 'NorthWest', 'SouthEast', 'SouthSouth', 'SouthWest', 'NorthCentral'])
state = st.selectbox("State", ['Adamawa', 'Bauchi', 'Borno', 'Gombe', 'Taraba', 'Yobe', 
                               'Kaduna', 'Katsina', 'Kano', 'Kebbi', 'Sokoto', 'Jigawa', 'Zamfara',
                               'Abia', 'Anambra', 'Ebonyi', 'Enugu', 'Imo',
                               'Akwa-Ibom', 'Bayelsa', 'Cross-River', 'Delta', 'Edo', 'Rivers',
                               'Ekiti', 'Lagos', 'Osun', 'Ondo', 'Ogun', 'Oyo',
                               'Benue', 'FCT', 'Kogi', 'Kwara', 'Nasarawa', 'Niger', 'Plateau'])
some_other_feature = st.number_input("Some Other Feature", min_value=0, value=0)  # Add this missing feature

# Mapping user inputs to the model's expected input format
input_data = [
    age,
    1 if gender == "Male" else 0,  # Assuming Male = 1, Female = 0
    1 if marital_status == "Married" else 0,  # Example encoding for marital status
    0,  # PolicyType (you need to map it based on your training data encoding)
    premium,
    num_claims,
    claim_satisfaction,
    0,  # PaymentHistory (map it based on how it was encoded in training)
    0,  # Product (map it based on training)
    1 if renewal_status == "Yes" else 0,
    1 if customer_category == "Individual" else 0,
    0,  # Region (map it based on encoding in the training dataset)
    0,  # State (map based on how it was encoded)
    some_other_feature  # Ensure the missing 14th feature is included here
]

# Predict churn when the user clicks the button
if st.button("Predict Churn"):
    prediction, churn_probability = predict_churn(input_data)
    
    if prediction == 1:
        st.error(f"The customer is likely to churn. Probability: {churn_probability:.2f}")
    else:
        st.success(f"The customer is not likely to churn. Probability: {1 - churn_probability:.2f}")

