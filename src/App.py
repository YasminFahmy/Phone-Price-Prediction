import streamlit as st
import os 
import numpy as np 
import pandas as pd
import joblib

st.set_page_config(
    page_title='hello',
    page_icon='',
    layout='centered',
    initial_sidebar_state='expanded'
)
def loaddate():
    scaler= joblib.load('../workspace/scaler.pkl')
    model= joblib.load('../workspace/model.pkl')
    model.feature_names=['weight','resoloution','ppi','cpu core','cpu freq','internal mem','ram','RearCam','Front_Cam','battery','thickness']
    return scaler, model

def insights():
    st.title('Phone price prediction')
    df=pd.read_csv('../datasets/Cellphone.csv')
    st.write(df.describe())

def main():
    st.title('Phone Price Prediction')
    input= {}
    scaler, model = loaddate()
    input['weight']=st.number_input('Enter weight', min_value=0, value=100)
    input['resoloution']=st.number_input('Enter resoloution', min_value=0, value=100)
    input['ppi']=st.number_input('Enter id', min_value=0, value=0)
    input['cpu core']=st.number_input('Enter cpu core', min_value=0, value=0)
    input['cpu freq']=st.number_input('Enter cpu freq', min_value=0, value=100)
    input['internal mem']=st.number_input('Enter internal mem', min_value=0, value=0)
    input['ram']=st.number_input('Enter ram', min_value=0, value=0)
    
    input['RearCam']=st.number_input('Enter RearCam', min_value=0, value=0)
    input['Front_Cam']=st.number_input('Enter Front_Cam', min_value=0, value=0)
    input['battery']=st.number_input('Enter battery', min_value=0, value=0)
    input['thickness']=st.number_input('Enter thickness', min_value=0, value=0)

    #weight	resoloution	ppi	cpu core	cpu freq	internal mem	ram	RearCam	Front_Cam	battery	thickness	Price
    in_df=pd.DataFrame([input])
    numerical=['weight','resoloution','ppi','cpu core','cpu freq','internal mem','ram','RearCam','Front_Cam','battery','thickness']
    in_df[numerical]=scaler.transform(in_df[numerical])

    in_df=in_df.reindex(columns=model.feature_names,  fill_value=0)
    if st.button('Predict price'):
        prediction=model.predict(in_df)
        st.success(f'Price would be: {prediction[0]:.2f}')
    # if(st.checkbox('insights')):
    #     insights()

if __name__ == '__main__':
    main()