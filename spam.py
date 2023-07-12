# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 02:19:49 2023

@author: golden deo
"""

import pickle
import streamlit as st
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

# Loading the saved model
loaded_model = pickle.load(open('spam_mail_prediction_model.sav', 'rb'))

# Creating a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Loading the vocabulary and IDF values
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Creating a function for prediction
def spam_email_model(input_mail):
    # Transforming the input_mail using the vectorizer
    input_data_features = vectorizer.transform(input_mail)

    prediction = loaded_model.predict(input_data_features)
    if prediction[0] == 0:
        return 'The mail is spam'
    else:
        return 'The mail is ham'

def main():
    # Giving a title
    st.title('Spam Email Check')
    
    # Getting the input data from the user
    message = st.text_input('Enter Your Mail')
    
    # Code for prediction
    diagnosis = ''
    
    # Creating a button for prediction
    if st.button('Spam Test Result'):
        diagnosis = spam_email_model([message])
        
    st.success(diagnosis)

if __name__ == '__main__':
    main()

# Loading the saved model
loaded_model = pickle.load(open('C:/Users/golden deo/OneDrive/Desktop/spam/spam_mail_prediction_model.sav', 'rb'))
