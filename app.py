import streamlit as st 
import pandas as pd
import joblib

model = joblib.load("model.pkl")

st.title("Employee Churn Predictor")

st.write("Predict whether an employee is likely to leave the company")

col1, col2, col3 = st.columns(3)

with col1:
    satisfaction_level = st.number_input("Satisfaction Level", min_value=0.0, max_value=1.0)
    average_montly_hours = st.number_input("Monthly WorkHours", min_value=0, max_value=650)
    promotion_last_5years = st.selectbox("Promotion Last 5 Years ?", ["Yes", "No"])

with col2:
    last_evaluation = st.number_input("Last Evaluation", min_value=0.0, max_value=1.0)
    time_spend_company = st.number_input("Working Years", min_value=0, max_value=30)
    salary = st.selectbox("salary", ["low", "medium", "high"])


with col3:
    number_project = st.number_input("Project Amount", min_value=0, max_value=15)
    Work_accident = st.selectbox("Work Accident ?", ["Yes", "No"])
    Department = st.selectbox("Department", ['sales', 'technical', 'support', 'IT', 'RandD', 'product_mng',
                                              'marketing', 'accounting', 'hr', 'management'])


if st.button('Predict'):
    input_data = pd.DataFrame({
        "satisfaction_level" : [satisfaction_level],
        "last_evaluation" : [last_evaluation],
        "number_project" : [number_project],
        "average_montly_hours" : [average_montly_hours],
        "time_spend_company" : [time_spend_company],
        "Work_accident" : [1 if (Work_accident=="Yes") else 0],
        "promotion_last_5years" : [1 if (promotion_last_5years=="Yes") else 0],
        "Department" : [Department],
        "salary" : [salary]
    })

    prediction = model.predict(input_data)[0]


    if prediction == 1:
        st.error("Employee is likely to turnover!")
    else:
        st.success("Employee is likely to stay.")