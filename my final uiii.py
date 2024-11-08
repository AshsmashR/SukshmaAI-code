import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder, RobustScaler
import cv2
import pickle
from tensorflow.keras.models import load_model

# Paths to the models, scaler, and encoder
stacking_model_path = r'C:\Users\Paru\OneDrive\Documents\intent\stacking_model.pkl'
scaler_path = r'C:\Users\Paru\OneDrive\Documents\intent\myscaler.pkl'
cnn_model_path = 'Pccnn_model.keras'
encoder_path = 'label_encoder.pkl'

# Load models, scaler, and label encoder
stacking_model = pickle.load(open(stacking_model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))
cnn_model = load_model(cnn_model_path)
with open(encoder_path, 'rb') as file:
    le = pickle.load(file)  # Load the same encoder used in training

# Set image size for CNN model
image_size = (128, 128)

# Define the feature inputs for stacking model
def get_user_input():
    user_data = {
        ' Age (yrs)': st.number_input("Age (yrs)", min_value=0, value=25),
        'Weight (Kg)': st.number_input("Weight (Kg)", min_value=0, value=60),
        'Height(Cm) ': st.number_input("Height(Cm)", min_value=0, value=160),
        'BMI': st.number_input("BMI", min_value=0.0, value=22.0),
        'Pulse rate(bpm) ': st.number_input("Pulse rate(bpm)", min_value=0, value=70),
        'Cycle length(days)': st.number_input("Cycle length(days)", min_value=0, value=28),
        'No. of abortions': st.number_input("No. of abortions", min_value=0, value=0),
        '1beta-HCG(mIU/mL)': st.number_input("1beta-HCG(mIU/mL)", min_value=0.0, value=5.0),
        '2beta-HCG(mIU/mL)': st.number_input("2beta-HCG(mIU/mL)", min_value=0.0, value=5.0),
        'FSH(mIU/mL)': st.number_input("FSH(mIU/mL)", min_value=0.0, value=5.0),
        'LH(mIU/mL)': st.number_input("LH(mIU/mL)", min_value=0.0, value=5.0),
        'FSH/LH': st.number_input("FSH/LH", min_value=0.0, value=1.0),
        'Waist(inch)': st.number_input("Waist(inch)", min_value=0, value=30),
        'Waist:Hip Ratio': st.number_input("Waist:Hip Ratio", min_value=0.0, value=0.8),
        'TSH (mIU/L)': st.number_input("TSH (mIU/L)", min_value=0.0, value=1.5),
        'AMH(ng/mL)': st.number_input("AMH(ng/mL)", min_value=0.0, value=3.0),
        'Vit D3 (ng/mL)': st.number_input("Vit D3 (ng/mL)", min_value=0.0, value=30.0),
        'Weight gain(Y/N)': st.selectbox("Weight gain(Y/N)", ["Y", "N"]),
        'Follicle No. (L)': st.number_input("Follicle No. (L)", min_value=0, value=10),
        'Follicle No. (R)': st.number_input("Follicle No. (R)", min_value=0, value=10),
        'Avg. F size (L) (mm)': st.number_input("Avg. F size (L) (mm)", min_value=0.0, value=5.0),
        'Avg. F size (R) (mm)': st.number_input("Avg. F size (R) (mm)", min_value=0.0, value=5.0),
        'Endometrium (mm)': st.number_input("Endometrium (mm)", min_value=0.0, value=10.0)
    }
    return pd.DataFrame(user_data, index=[0])

# Encode categorical features
def encode_categorical(df):
    encoder_dict = {'Weight gain(Y/N)': ['N', 'Y']}
    le_cat = LabelEncoder()
    for col, classes in encoder_dict.items():
        le_cat.classes_ = np.array(classes)
        df[col] = le_cat.transform(df[col])
    return df

# Load and preprocess the image for CNN prediction
def preprocess_image_for_cnn(uploaded_file):
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, image_size).astype('float32') / 255.0
    image = np.expand_dims(image, axis=(0, -1))  # Add batch and channel dimensions
    return image

# Main function to run the app
def main():
    # App Header and Instructions
    st.title("PCOS Prediction App")
    st.image("https://via.placeholder.com/800x200.png?text=PCOS+Prediction+App", use_column_width=True)  # Header image placeholder
    st.markdown("""
    Welcome to the **PCOS Prediction App**!  
    This tool helps predict the likelihood of PCOS using two models:
    - **Stacking Model**: Uses clinical data features for prediction.
    - **CNN Model**: Uses ultrasound images for prediction.
    
    Choose a model from the sidebar and provide the necessary input.
    """)

    st.sidebar.header("🩺 Select Prediction Model")
    model_choice = st.sidebar.radio("Choose the model for prediction:", ("Stacking Model", "CNN Model"))

    # Display input summary
    st.sidebar.header("📋 Selected Inputs Summary")
    user_input_df = get_user_input()
    st.sidebar.write(user_input_df)

    # Display UI based on selected model
    if model_choice == "Stacking Model":
        st.subheader("PCOS Prediction using Stacking Model")

        # Encode and scale features for stacking model
        encoded_df = encode_categorical(user_input_df)
        scaled_features = scaler.transform(encoded_df)

        if st.button("Predict with Stacking Model"):
            with st.spinner('Making prediction...'):
                prediction = stacking_model.predict(scaled_features)
                confidence = np.max(stacking_model.predict_proba(scaled_features)) * 100  # Model confidence

                # Interpret and display the result
                result = "PCOS Detected" if prediction[0] == 1 else "PCOS Not Detected"
                st.success(f'The prediction is: {result}')
                st.info(f"Model confidence: {confidence:.2f}%")
                st.progress(int(confidence))  # Display confidence as a progress bar

            # Download report
            result_text = f"The prediction is: {result}\nConfidence: {confidence:.2f}%"
            st.download_button(
                label="Download Report",
                data=result_text,
                file_name="PCOS_Prediction.txt",
                mime="text/plain"
            )

    elif model_choice == "CNN Model":
        st.subheader("PCOS Prediction using CNN Model with Ultrasound Image")

        uploaded_file = st.file_uploader("Upload an ultrasound image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Show preview
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

            # Rotate or enhance image buttons
            if st.button("Rotate 90°"):
                image = preprocess_image_for_cnn(uploaded_file).squeeze()
                rotated_image = np.rot90(image).astype('float32')
                st.image(rotated_image, caption="Rotated Image", use_column_width=True)

            if st.button("Enhance Contrast"):
                image = preprocess_image_for_cnn(uploaded_file).squeeze()
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_image = clahe.apply((image * 255).astype(np.uint8))
                st.image(enhanced_image, caption="Enhanced Image", use_column_width=True)

            # Preprocess image and make prediction
            image_data = preprocess_image_for_cnn(uploaded_file)
            with st.spinner('Making prediction...'):
                prediction = cnn_model.predict(image_data)
                confidence = np.max(prediction) * 100  # Model confidence
                predicted_class_name = le.inverse_transform([np.argmax(prediction)])[0]  # Use LabelEncoder directly

                st.success(f"The prediction is: {predicted_class_name}")
                st.info(f"Model confidence: {confidence:.2f}%")
                st.progress(int(confidence))  # Display confidence as a progress bar

    # Add final info message
    st.info("This application is designed to assist with PCOS prediction. Please select the model and provide the required inputs.")

    # Styling for visual enhancement
    st.markdown("""
    <style>
    .stButton>button {background-color: #4CAF50; color: white;}
    .stProgress>div>div {background-color: #4CAF50;}
    .stSidebar {background-color: #f0f2f6;}
    </style>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
