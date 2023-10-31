import streamlit as st
import pickle as pk
from numpy import array
from sklearn.ensemble import RandomForestRegressor
import base64

with open("model.pkl","rb") as file:
    rfr_model=pk.load(file)

st.title("GDP Prediction of a country")

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    backdrop-filter: blur(1000px);
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background(r"coronavirus-infection-background-with-copy-space.jpg")

st.markdown("----")

if rfr_model is not None:
    st.header("Input Data")

    overall = st.number_input("Enter overall economic activity from each sector:", min_value=0)
    employment = st.number_input("Enter employment (in thousands):", min_value=0)

    if st.button("Predict GDP"):
        # Create an input array
        input_data = array([[overall, employment]])

        prediction = rfr_model.predict(input_data)[0]

        st.header("Prediction Result")
        st.write(f"Predicted GDP: ${prediction:,.2f}")

    st.sidebar.header("App Customization")

    new_bg_image = st.sidebar.file_uploader("Change Background Image (PNG or JPG)", type=["png", "jpg"])
    if new_bg_image:
        with open("custom_bg.png", "wb") as f:
            f.write(new_bg_image.read())
        set_background("custom_bg.png")

    st.sidebar.header("About")
    st.sidebar.write("This app predicts a country's GDP for year 2021 based on user-provided data.")
    st.sidebar.write("It takes user input as 'Total Economy(in trillions)' and 'Employment(in thousands)'")
    st.sidebar.write("Model: Random Forest Regressor")

else:
    st.write("Model failed to load. Please check the model file path.")