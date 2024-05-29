import streamlit as st
import pandas as pd
import joblib

def model_page():
    st.title("Model Prediksi Jamur Beracun atau Tidak")
    st.write("Memprediksi apakah jamur edible atau beracun berdasarkan fitur pada data")
    st.sidebar.header('User Input Features')

    input_data = user_input()

    st.subheader('User Input')
    st.write(input_data)

    load_model = joblib.load("KNN_best.pkl")

    prediction = load_model.predict(input_data)

    if prediction == 1:
        prediction = 'Jamur beracun'
    else:
        prediction = 'Jamur edible'

    st.write('Based on user input, the model predicted: ')
    st.write(prediction)

def user_input():
    cap_diameter = st.sidebar.number_input('cap-diameter', min_value=0, max_value=2000, value=0)
    cap_shape = st.sidebar.selectbox('cap-shape', [0, 1, 2, 3, 4, 5, 6])
    gill_attachment = st.sidebar.selectbox('gill-attachment', [0, 1, 2, 3, 4, 5, 6])
    gill_color = st.sidebar.selectbox('gill-color', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    stem_height = st.sidebar.number_input('stem-height', min_value=0, max_value=4, value=0)
    stem_width = st.sidebar.number_input('stem-width', min_value=0, max_value=4000, value=0)
    stem_color = st.sidebar.selectbox('stem-color', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    season = st.sidebar.selectbox('season', [1.80427271, 0.94319455, 0.88845029, 0.02737213])

    data = {
        'cap-diameter': cap_diameter,
        'cap-shape': cap_shape,
        'gill-attachment': gill_attachment,
        'gill-color': gill_color,
        'stem-height': stem_height,
        'stem-width': stem_width,
        'stem-color': stem_color,
        'season': season
    }

    features = pd.DataFrame(data, index=[0])
    return features

if __name__ == '__main__':
    model_page()
