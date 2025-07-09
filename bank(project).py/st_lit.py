import streamlit as st
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from tensorflow.keras.models import load_model
import joblib

#python -m streamlit run st_lit.py

model=load_model('my_bank_model.h5')
st.title('Credit Risk Prediction 📊 ')

Duration_of_Credit_Credit_monthly=st.number_input('Duration_Of_Credit_Credit_Monthly',step=1.0)
Credit_Amount=st.number_input('Credit_Amount',step=1.0)
Age_years=st.number_input('Age_years',step=1.0)
Duration_in_Current_address=st.number_input('Duration_in_Current_address',step=1.0)

Account_Balance=st.selectbox('Account_Balance',
                             ['No_Current_Account_Found',
                             'Less_than_200_DM',
                             'From_200_To_999_DM',
                           'More_than_1000_DM'])

Purpose=st.selectbox('Purpose',
    ['unknown',
     'New_car',
     'Used_car',
     'Furniture',
     'Home_appliances',
     'Tv_or_Radio',
     'Tuition_fees',
     'Vacation',
     'personal_reason','Ather'])

Value_Savings_Stocks=st.selectbox('Value_Savings_Stocks',
    ['less_than_100_DM',
    'From_100_To_499_DM',
    'Form_500_To_999_DM',
    'More_than_1000_DM',
    'Unknown'])

Length_of_current_employment=st.selectbox('Length_of_current_employment',
    ['Less_than_year',
     'From_1year_to_4years',
     'From_4years_to_7years',
     'More_than_7years',
     'Unemployed'])

Instalment_per_cent=st.selectbox('Instalment_per_cent',
    ['20% of income',
     '25% of income',
     '30% of income',
     '35% of income'])


Sex_Marital_Status=st.selectbox('Sex_Marital_Status',
    ['Single_male',
     'Married_female',
     'Married_male','Divorced'])


Guarantors=st.selectbox('Guarantors',
    ['None',
      'Partner',
     'Other Guarantor'])


Most_valuable_available_asset=st.selectbox('Most_valuable_available_asset',
   ['car',
     'Property_ownership',
      'Life_Insurance',
     'Nothing'])


Concurrent_Credits=st.selectbox('Concurrent_Credits',
                                ['Nothing',
                                 'Store',
                                 'Anather_bank'])


Type_of_apartment=st.selectbox('Type_of_apartment',['Rent'
                                                    ,'Joint_ownership',
                                                    'Private_ownership'])


Occupation=st.selectbox('Occupation',['Unemployed'
                                      ,'simple_employe',
                                      'Good employee',
                                      'Highly paid worker'])

Payment_Status_of_Previous_Credit=st.selectbox('Payment_Status_of_Previous_Credit',['No Credit',
                                                                                    'Paid on Time',
                                                                                    'Paid in Full',
                                                                                    'Slight Delay',
                                                                                    'Critical / Default'])

Telephone=st.selectbox('Telephone',['Yes','NO'])
Foreign_Worker=st.selectbox('Foreign_Worker',['Yes','No'])


input_dict={
    'Account_Balance':Account_Balance,
    'Duration_of_Credit_Credit_monthly':Duration_of_Credit_Credit_monthly,
    'Payment_Status_of_Previous_Credit':Payment_Status_of_Previous_Credit,
    'Purpose':Purpose,
    'Credit_Amount' :Credit_Amount,
    'Value_Savings_Stocks':Value_Savings_Stocks,
    'Length_of_current_employment':Length_of_current_employment,
    'Instalment_per_cent': Instalment_per_cent,
    'Sex_Marital_Status':Sex_Marital_Status,
    'Guarantors':Guarantors,
    'Duration_in_Current_address':Duration_in_Current_address,
    'Most_valuable_available_asset':Most_valuable_available_asset,
    'Age_years':Age_years,
    'Concurrent_Credits':Concurrent_Credits,
    'Type_of_apartment':Type_of_apartment,
    'Occupation':Occupation,
    'Telephone':Telephone,
    'Foreign_Worker':Foreign_Worker
}

df_input=pd.DataFrame([input_dict])

df_cat = df_input.select_dtypes(include='object')
df_num = df_input.select_dtypes(exclude='object')

encoder=joblib.load('encoder.joblib')
encoded_array=encoder.transform(df_cat)
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(df_cat.columns))
encoded_df.index=df_input.index
df_final = pd.concat([encoded_df, df_num], axis=1)

prediction=model.predict(df_final)[0][0]

st.subheader('Resalt')
btn=st.button('Predict')
if btn:
   if prediction >=0.5:
      st.success(' Qualified ✅ ')
   else:
      st.error('Unqualified ❌ ')