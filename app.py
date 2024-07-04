import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

def main():
    st.title("Customer Churn Prediction")

    st.header("Customer Information")

    gender = st.selectbox("Gender", ['Male', 'Female'])
    gender = 1 if gender == 'Male' else 0

    senior_citizen = st.selectbox("Senior Citizen", ['Yes', 'No'])
    senior_citizen = 1 if senior_citizen == 'Yes' else 0

    partner = st.selectbox("Partner", ['Yes', 'No'])
    partner = 1 if partner == 'Yes' else 0

    dependents = st.selectbox("Dependents", ['Yes', 'No'])
    dependents = 1 if dependents == 'Yes' else 0

    tenure = st.number_input("Tenure (months)", min_value=0)

    st.header("Services and Charges")

    phone_service = st.selectbox("Phone Service", ['Yes', 'No'])
    phone_service = 1 if phone_service == 'Yes' else 0

    multiple_lines = st.selectbox("Multiple Lines", ['Yes', 'No'])
    multiple_lines = 1 if multiple_lines == 'Yes' else 0

    online_security = st.selectbox("Online Security", ['Yes', 'No'])
    online_security = 1 if online_security == 'Yes' else 0

    online_backup = st.selectbox("Online Backup", ['Yes', 'No'])
    online_backup = 1 if online_backup == 'Yes' else 0

    device_protection = st.selectbox("Device Protection", ['Yes', 'No'])
    device_protection = 1 if device_protection == 'Yes' else 0

    tech_support = st.selectbox("Tech Support", ['Yes', 'No'])
    tech_support = 1 if tech_support == 'Yes' else 0

    streaming_tv = st.selectbox("Streaming TV", ['Yes', 'No'])
    streaming_tv = 1 if streaming_tv == 'Yes' else 0

    streaming_movies = st.selectbox("Streaming Movies", ['Yes', 'No'])
    streaming_movies = 1 if streaming_movies == 'Yes' else 0

    paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
    paperless_billing = 1 if paperless_billing == 'Yes' else 0

    st.header("Charges")

    monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
    total_charges = st.number_input("Total Charges", min_value=0.0)

    st.header("Internet Service")

    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    internet_service_dsl = 1 if internet_service == 'DSL' else 0
    internet_service_fiber = 1 if internet_service == 'Fiber optic' else 0
    internet_service_no = 1 if internet_service == 'No' else 0

    st.header("Contract")

    contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    contract_month_to_month = 1 if contract == 'Month-to-month' else 0
    contract_one_year = 1 if contract == 'One year' else 0
    contract_two_year = 1 if contract == 'Two year' else 0

    st.header("Payment Method")

    payment_method = st.selectbox("Payment Method", ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'])
    payment_method_bank_transfer = 1 if payment_method == 'Bank transfer (automatic)' else 0
    payment_method_credit_card = 1 if payment_method == 'Credit card (automatic)' else 0
    payment_method_electronic_check = 1 if payment_method == 'Electronic check' else 0
    payment_method_mailed_check = 1 if payment_method == 'Mailed check' else 0

    if st.button("Submit"):
        # Apply MinMaxScaler to the charges
        scaler = MinMaxScaler()
        charges = np.array([[monthly_charges, total_charges]])
        scaled_charges = scaler.fit_transform(charges)

        data = {
            'gender': [gender],
            'SeniorCitizen': [senior_citizen],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'PaperlessBilling': [paperless_billing],
            'MonthlyCharges': [scaled_charges[0][0]],
            'TotalCharges': [scaled_charges[0][1]],
            'InternetService_DSL': [internet_service_dsl],
            'InternetService_Fiber optic': [internet_service_fiber],
            'InternetService_No': [internet_service_no],
            'Contract_Month-to-month': [contract_month_to_month],
            'Contract_One year': [contract_one_year],
            'Contract_Two year': [contract_two_year],
            'PaymentMethod_Bank transfer (automatic)': [payment_method_bank_transfer],
            'PaymentMethod_Credit card (automatic)': [payment_method_credit_card],
            'PaymentMethod_Electronic check': [payment_method_electronic_check],
            'PaymentMethod_Mailed check': [payment_method_mailed_check]
        }

        df = pd.DataFrame(data, columns=[
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
            'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
            'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 
            'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No', 
            'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year', 
            'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)', 
            'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
        ])


        # Load the pre-trained model
        model = load_model('best_model.h5')

        # Make prediction
        prediction = model.predict(df)
        prediction_result = (prediction > 0.5).astype(int)

        if prediction_result[0][0] == 1:
            st.write("Prediction: The customer is likely to churn.")
        else:
            st.write("Prediction: The customer is not likely to churn.")



if __name__ == '__main__':
    main()