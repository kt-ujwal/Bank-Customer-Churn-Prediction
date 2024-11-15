import streamlit as st
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
import os
from openai import OpenAI
from utils import create_gauge_chart, create_mode_probaibility_chart

client = OpenAI(base_url="https://api.groq.com/openai/v1",
                api_key=os.environ["GROQ_KEY"])


def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


xgb_model_fe = load_model('xgb_model_fe.pkl')
xgb_model_smote = load_model('xgb_model_smote.pkl')
voting_clf = load_model('voting_clf_soft.pkl')


def prepare_input(credit_score, location, gender, age, tenure, balance,
                  num_products, has_credit_card, is_active_member,
                  estimated_salary):
    clv = balance * estimated_salary / 100000
    tenure_age_ratio = tenure / age if age > 0 else 0

    if age <= 30:
        age_group = 'Young'
    elif age <= 45:
        age_group = 'MiddleAge'
    elif age <= 60:
        age_group = 'Senior'
    else:
        age_group = 'Elderly'

    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': has_credit_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary,
        'Geography_France': 1 if location == 'France' else 0,
        'Geography_Germany': 1 if location == 'Germany' else 0,
        'Geography_Spain': 1 if location == 'Spain' else 0,
        'Gender_Female': 1 if gender == 'Female' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'CLV': clv,
        'TenureAgeRatio': tenure_age_ratio,
        'AgeGroup_MiddleAge': 1 if age_group == 'MiddleAge' else 0,
        'AgeGroup_Senior': 1 if age_group == 'Senior' else 0,
        'AgeGroup_Elderly': 1 if age_group == 'Elderly' else 0,
    }

    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict


def predict(input_df, input_dict):
    probabilities = {
        'XGBoost': xgb_model_fe.predict_proba(input_df)[0][1],
        'XGBoost_SMOTE': xgb_model_smote.predict_proba(input_df)[0][1],
        'Voting_CLF': voting_clf.predict_proba(input_df)[0][1],
    }

    avg_probability = np.mean(list(probabilities.values()))
    col1, col2 = st.columns(2)
    with col1:
        fig = create_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True)
        st.write(
            f"The customer has a {avg_probability:.2%} probability of churning."
        )
    with col2:
        figs_prob = create_mode_probaibility_chart(probabilities)
        st.plotly_chart(figs_prob, use_container_width=True)

    return avg_probability


def explain_prediction(probability, input_dict, surname):
    prompt = f"""You are an expert data scientist at a bank, where you specialize in interpreting and explaining predicitons of machine learning models.
    
    Your machine learning model has predicted that a customer named {surname} has a {round(probability*100,1)} probability of churning, based on the information provided below.
    
    Here is the customer's information:
    {input_dict}
    Here are the machine learning model's top 12 most important features for predicting churn:

        Feature          | Importance
    ____________________________________
    NumOfProducts        | 0.323888
    IsActiveMember       | 0.164146
    Age                  | 0.109550
    Geography_Germany    | 0.091373
    Balance              | 0.052786
    Geography_France     | 0.046463
    Gender_Female        | 0.036855
    Geography_Spain      | 0.036855
    CreditScore          | 0.035005
    EstimatedSalary      | 0.032655
    HasCrCard            | 0.031940
    Tenure               | 0.030054

    {pd.set_option('display.max_colwidth', None)}

    Here are the summary statistics for churned customers:
    {df[df['Exited']==1].describe()}

    - If the customer has over a 40% risk of churning, generate a 3 sentence explanation of why they are at a risk of churning.
    - If the customer has less than a 40% risk of churning, generate a 3 sentence explanation of why they are not at a risk of churn
    - Your explanation should be based on the customer's information, the summary statistics of churned and non-churned cusotmers and the model's top 12 most important features.".

    Don't mention the probabiltiy of churning, or the machine learning model, or say anything like "based on the model's predictions and top 12 most important features". Make sure to keep it human readable and don't directly print the exact statistic value.

    
    """
    print("EXPLANATION PROMPT", prompt)
    raw_response = client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )
    return raw_response.choices[0].message.content


def generate_email(probablity, input_dict, explanation, surname):
    prompt = f"""Your are John Doe, a manager at EU Bank. you are responsible for ensuring cutomers stay with the bank and are incentivized with various offers.
    You notice a customer named {surname} has a {round(probablity*100,1)}% probabiltiy of churning.
    Here is the customer's information:
    {input_dict}
    Here is some explantion as to why the customer might be at the risk of churning:
    {explanation}

    Generate a personalized email to the customer based on their information, asking them to stay if they are at the risk of churning, or offering them incentives so that they become more loyal to the bank.

    Make sure to list out a set of incentives to stay based on thier information, in bullet point format. Don't ever mention the probability of churning, or the machine learning model to the customer.
    """
    raw_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )
    print("\n\n EMAIL PROMPT", prompt)
    return raw_response.choices[0].message.content


st.title("Customer Churn Prediction")

df = pd.read_csv("churn.csv")
customers = [
    f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()
]
selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
    selected_customer_id = int(selected_customer_option.split(" - ")[0])
    selected_customer = df.loc[df['CustomerId'] == selected_customer_id]

    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input("Credit Score",
                                       min_value=300,
                                       max_value=850,
                                       value=int(
                                           selected_customer['CreditScore']))
        location = st.selectbox("Location", ["Spain", "France", "Germany"],
                                index=["Spain", "France", "Germany"].index(
                                    selected_customer['Geography'].iloc[0]))
        gender = st.radio(
            "Gender", ["Male", "Female"],
            index=0 if selected_customer['Gender'].iloc[0] == 'Male' else 1)
        age = st.number_input("Age",
                              min_value=18,
                              max_value=100,
                              value=int(selected_customer['Age']))
        tenure = st.number_input("Tenure (years)",
                                 min_value=0,
                                 max_value=50,
                                 value=int(selected_customer['Tenure']))

    with col2:
        balance = st.number_input("Balance",
                                  min_value=0.0,
                                  value=float(selected_customer['Balance']))
        num_products = st.number_input("Number of Products",
                                       min_value=1,
                                       max_value=10,
                                       value=int(
                                           selected_customer['NumOfProducts']))
        has_credit_card = st.checkbox(
            "Has Credit Card",
            value=bool(selected_customer['HasCrCard'].iloc[0]))
        is_active_member = st.checkbox(
            "Is Active Member",
            value=bool(selected_customer['IsActiveMember'].iloc[0]))
        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=float(selected_customer['EstimatedSalary']))

    input_df, input_dict = prepare_input(credit_score, location, gender, age,
                                         tenure, balance, num_products,
                                         has_credit_card, is_active_member,
                                         estimated_salary)
    avg_probability = predict(input_df, input_dict)
    explanation = explain_prediction(avg_probability, input_dict,
                                     selected_customer['Surname'])
    st.markdown("---")
    st.markdown("###  Explanation of Prediction")
    st.markdown(explanation)
    email = generate_email(avg_probability, input_dict, explanation,
                           selected_customer['Surname'])
    st.markdown("---")
    st.markdown("###  Personalized Email")
    st.markdown(email)
