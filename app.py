import streamlit as st
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from feature_extraction import extract_features

st.set_page_config(page_title="Plant Stress Analysis", layout="wide")
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}

h1 {
    color: #ffffff;
    text-align: center;
    font-weight: bold;
}

[data-testid="stSidebar"] {
    background-color: #1b2b34;
}

div.stButton > button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)
st.title("🌱 Plant Stress Analysis Using Machine Learning")

# Load trained model
model = pickle.load(open("model.pkl", "rb"))
uploaded_file = st.file_uploader("Upload Plant Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Image", width=400)

    features = extract_features(img)
    prediction = model.predict([features])[0]

    mean_intensity, green_ratio, texture = features

    with col2:
        st.subheader("Prediction Result")

        if prediction == 0:
            st.success("🌿 Plant is Healthy")
        else:
            st.error("⚠ Plant is Stressed")

        st.subheader("Feature Analysis")

        st.write(f"Mean Intensity: {mean_intensity:.2f}")
        st.write(f"Green Ratio: {green_ratio:.2f}")
        st.write(f"Texture Value: {texture:.2f}")

        st.subheader("Explanation")

        if green_ratio < 0.3:
            st.write("✔ Low Green Ratio → Chlorophyll reduction detected")

        if texture > 100:
            st.write("✔ High Texture → Possible leaf surface damage")

        if mean_intensity < 80:
            st.write("✔ Low Intensity → Possible water stress")

        st.write("Final conclusion based on extracted features and ML prediction.")

    st.subheader("Feature Visualization")

    fig, ax = plt.subplots()
    ax.bar(["Intensity", "Green Ratio", "Texture"], features)

    st.pyplot(fig)
