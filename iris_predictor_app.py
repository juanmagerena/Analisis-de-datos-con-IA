
import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Iris Species Predictor", layout="centered")
st.title('🌸 Iris Species Predictor')
st.write('Enter the measurements of the Iris flower to predict its species.')

# Define a function to load the model and label encoder
# This is important because Streamlit re-runs the script, and we want to cache resources
@st.cache_resource
def load_resources():
    # Ensure we are in the correct directory to load the pkl files
    # The directory was set in a previous step, but explicitly setting it here for robustness
    expected_dir = '/content/drive/MyDrive/Ejercicios IA/Despliegue'
    current_dir_at_load = os.getcwd()
    if current_dir_at_load != expected_dir:
        try:
            os.chdir(expected_dir)
            print(f"[Streamlit App] Changed working directory to: {os.getcwd()}")
        except Exception as e:
            st.error(f"[Streamlit App] Could not change directory to {expected_dir}: {e}")
            st.stop()
    
    try:
        model = 'best_knn_model.pkl'
        label_encoder = 'label_encoder_iris_species.pkl'
    except FileNotFoundError as e:
        st.error(f"Model or label encoder files not found: {e}. Make sure 'best_knn_model.pkl' and 'label_encoder_iris_species.pkl' are in {expected_dir}.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()
    return model, label_encoder

# Load the model and label encoder
best_knn_model, label_encoder_iris_species = load_resources()

# Input fields for flower features with user-friendly labels
st.subheader('Flower Measurements (cm)')
sepal_length = st.slider('Sepal Length', 0.0, 10.0, 5.0, 0.1)
sepal_width = st.slider('Sepal Width', 0.0, 10.0, 3.0, 0.1)
petal_length = st.slider('Petal Length', 0.0, 10.0, 4.0, 0.1)
petal_width = st.slider('Petal Width', 0.0, 10.0, 1.5, 0.1)

if st.button('Predict Species 🚀'):
    # Prepare the input features as a NumPy array
    flower_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Make a prediction using the loaded KNN model
    predicted_species_encoded = best_knn_model.predict(flower_features)

    # Attempt to decode the predicted species using the label encoder
    predicted_species_decoded = label_encoder_iris_species.inverse_transform(predicted_species_encoded)

    if isinstance(predicted_species_decoded[0], str):
        final_species_name = predicted_species_decoded[0]
    else:
        # If inverse_transform returned a numerical value, use a common mapping
        iris_species_mapping = {
            0: 'Iris-setosa',
            1: 'Iris-versicolor',
            2: 'Iris-virginica'
        }
        final_species_name = iris_species_mapping.get(predicted_species_encoded[0], "Unknown Species (Mapping not found)")
        st.warning("Note: The loaded label encoder did not return a string species name. Using a common Iris species mapping for display.")

    st.success(f"The predicted Iris species is: **{final_species_name}**")

st.markdown("""
--- 
*This app uses a K-Nearest Neighbors model to predict Iris species.*
""")
