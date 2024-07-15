import pickle
import streamlit as st

# Load the saved model
parkinson_model = pickle.load(open('C:/Users/admin/Music/parkinson_s prediction/model.sav', 'rb'))

# Home page
def home_page():
    st.title("Home Page")
    st.image("download.jpg")
    st.write("Welcome to the Parkinson's disease prediction based on the vocals dataset using machine learning ")

import pickle
import streamlit as st

# Load the saved model
parkinson_model = pickle.load(open('C:/Users/admin/Music/parkinson_s prediction/model.sav', 'rb'))

# Prediction page
import pickle
import streamlit as st

# Load the saved model
parkinson_model = pickle.load(open('C:/Users/admin/Music/parkinson_s prediction/model.sav', 'rb'))

# Prediction page
def prediction_page():
    st.title("Prediction Page")
    st.write("Enter the required values to predict Parkinson's disease:")
    col1, col2 = st.columns(2)

    with col1:
        MDVP_Fhi_Hz = st.text_input('MDVP:Fhi(Hz)', value='', help='Enter a value between 0 and 500')
        MDVP_Flo_Hz = st.text_input('MDVP:Flo(Hz)', value='', help='Enter a value greater than 0')
        MDVP_Jitter_abs = st.text_input('MDVP:Jitter(%)', value='', help='Enter a value greater than 0')
        MDVP_Jitter_RAP = st.text_input('MDVP:Jitter(RAP)', value='', help='Enter a value greater than 0')
        MDVP_Jitter_PPQ = st.text_input('MDVP:Jitter(PPQ)', value='', help='Enter a value greater than 0')
        MDVP_Shimmer = st.text_input('MDVP:Shimmer', value='', help='Enter a value greater than 0')
        MDVP_Shimmer_dB = st.text_input('MDVP:Shimmer(dB)', value='', help='Enter a value greater than 0')
        MDVP_Shimmer_APQ3 = st.text_input('MDVP:Shimmer(APQ3)', value='', help='Enter a value greater than 0')
        MDVP_Shimmer_APQ5 = st.text_input('MDVP:Shimmer(APQ5)', value='', help='Enter a value greater than 0')
        NHR = st.text_input('NHR', value='', help='Enter a value greater than 0')

    with col2:
        HNR = st.text_input('HNR', value='', help='Enter a value greater than 0')
        RPDE = st.text_input('RPDE', value='', help='Enter a value between 0 and 1')
        DFA = st.text_input('DFA', value='', help='Enter a value between 0 and 1')
        PPE = st.text_input('PPE', value='', help='Enter a value greater than 0')
        Jitter_DDP = st.text_input('Jitter:DDP', value='', help='Enter a value greater than 0')
        MDVP_APQ = st.text_input('MDVP:APQ', value='', help='Enter a value greater than 0')
        Shimmer_DDA = st.text_input('Shimmer:DDA', value='', help='Enter a value greater than 0')
        MDVP_DDA = st.text_input('MDVP:DDA', value='', help='Enter a value greater than 0')

    # Code for prediction
    parkinson_prediction = ''

    # Creating a button for Prediction
    if st.button('Parkinsons disease prediction Result'):
        # Convert input data to floats and handle empty strings
        input_data = [float(val) if val.strip() else 0.0 for val in [
            MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter_abs, MDVP_Jitter_RAP, MDVP_Jitter_PPQ,
            MDVP_Shimmer, MDVP_Shimmer_dB, MDVP_Shimmer_APQ3, MDVP_Shimmer_APQ5, NHR,
            HNR, RPDE, DFA, PPE, Jitter_DDP, MDVP_APQ, Shimmer_DDA, MDVP_DDA
        ]]

        # Check if any input values are negative
        if any(val < 0 for val in input_data):
            # Display an alert message if any value is negative
            st.warning("Please enter only positive values for all inputs.")
        else:
            # Make prediction using the reshaped input data
            parkinson_prediction = parkinson_model.predict([input_data])

            # Replace the labels for better interpretation
            if parkinson_prediction[0] == 0:
                prediction_label = "No Parkinson's Disease"
            else:
                prediction_label = "Parkinson's Disease Detected"

            st.success(f'Prediction: {prediction_label}')

            # Display recommendations based on prediction
            if parkinson_prediction[0] == 1:
                st.write("Recommendations:")
                st.write("- Consult with a neurologist for further evaluation.")
                st.write("- Regular monitoring of symptoms is advisable.")
                st.write("- Consider lifestyle changes for better management, such as regular exercise and a balanced diet.")
            else:
                st.write("No recommendations needed. However, regular health check-ups are advisable.")

# Streamlit app


# About page
def about_page():
    st.title("About")

    st.write("Welcome to our Parkinson's Disease Prediction web application!")
    st.write("# **About the Project**")
    st.write("""This project aims to predict Parkinson's disease based on vocal features using machine learning. 
    We utilize the Parkinson's disease dataset, which contains various vocal features collected from individuals with and without Parkinson's disease.""")
    st.write("# **Key Objectives**")
    st.write("### **Early Diagnosis**")
    st.write("Our primary objective is to develop a model that can accurately diagnose Parkinson's disease based on vocal features. Early diagnosis can lead to better management and treatment of the disease.")
    st.write("### **Machine Learning Approach**")
    st.write("We leverage machine learning techniques, specifically RandomForestClassifier, to train a model on the dataset. This allows us to predict Parkinson's disease based on vocal biomarkers.")
    st.write("### **Web Application**")
    st.write("By deploying the trained model into a web application using Streamlit, we provide a user-friendly interface for predicting Parkinson's disease.")
    st.write("# **Get Involved**")
    st.write("Excited about the potential of using machine learning for healthcare? Join us on our journey as we explore innovative solutions in disease prediction and diagnosis. Engage with our community, share your expertise, and together, let's make a difference in healthcare.")

# Streamlit app
def main():
    # Set color and size for the sidebar title
    st.sidebar.markdown("<h1 style='color: blue; font-size: 24px;'>Parkinsons disease prediction based on the vocals dataset using machine learning</h1>", unsafe_allow_html=True)

    # Create the radio button
    page = st.sidebar.radio(" ", ["Home", "Prediction Page", "About"])

    if page == "Home":
        home_page()
    elif page == "Prediction Page":
        prediction_page()
    elif page == "About":
        about_page()

if __name__ == "__main__":
    main()
