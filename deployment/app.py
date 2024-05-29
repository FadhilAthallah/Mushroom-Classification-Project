import streamlit as st
import numpy as np
import pandas as pd
import joblib

from eda import eda_page
from prediction import model_page

# Load data
df = pd.read_csv("mushroom_cleaned.csv")

st.header('**MILESTONE 2**')
st.write("""
Created by Fadhil Athallah - HCK015 """)

st.write('Program ini dibuat untuk memprediksi jamur beracun atau tidak berdasarkan fitur-fitur yang terdapat dari dataset mushrrom_cleaned pada kaggle')
df

def main():
    # Define menu options
    menu_options = ["EDA", "Model"]

    # Create sidebar menu
    selected_option = st.sidebar.radio("Menu", menu_options)

    # Display selected page
    if selected_option == "EDA":
        eda_page()
    elif selected_option == "Model":
        model_page()


if __name__ == "__main__":
    main()
