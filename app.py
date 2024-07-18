import streamlit as st
import requests
from PIL import Image
import io

# FastAPI endpoint
FASTAPI_URL = "http://localhost:8080/predict"

# Page configuration
st.set_page_config(
    page_title="Potato Disease Classifier",
    page_icon="🥔",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    .st-emotion-cache-fis6aj e1b2p2ww10 {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background-color: #E8F5E9;
        height: 300px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .prediction-box {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        height: 300px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1 {
        color: #2E7D32;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🥔 Potato Disease Classifier")
st.markdown("<p style='text-align: center;'>Upload an image of a potato plant leaf to classify its disease.</p>", unsafe_allow_html=True)

# Create three columns
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    custom_css = """
        <style>
            .file-input {
                border: 2px dashed #ccc;
                padding: 20px;
                text-align: center;
            }
            .file-input:hover {
                border-color: #555;
            }
        </style>
    """

    # Display the custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


    if uploaded_file is not None:
        if st.button("Classify Disease", key='classify_button'):
            files = {"file": ("image.jpg", uploaded_file.getvalue(), "image/jpeg")}
            with st.spinner("Analyzing image..."):
                response = requests.post(FASTAPI_URL, files=files)
                if response.status_code == 200:
                    result = response.json()
                    with col3:
                        st.markdown(f"""
                        <div class='prediction-box'>
                        <h2 style='text-align: center; color: #4CAF50;'>Prediction Result</h2>
                        <h3 style='text-align: center;'>Disease: {result['class']}</h3>
                        <h4 style='text-align: center;'>Confidence: {result['confidence']:.2f}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error("Error occurred during prediction. Please try again.")

with col2:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    else:
        st.markdown("""
        <div style='height: 300px; display: flex; justify-content: center; align-items: center;'>
        <p style='text-align: center; color: #666;'>Upload an image to see it here</p>
        </div>
        """, unsafe_allow_html=True)

with col3:
    if uploaded_file is None:
        st.markdown("""
        <div class='prediction-box'>
            <h2 style='text-align: center; color: #4CAF50;'>Prediction Result</h2>
            <p style='text-align: center;'>Upload an image and click 'Classify Disease' to see the results here.</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>This app uses a machine learning model to classify potato plant diseases.</p>
        <p>Supported diseases: Early Blight, Late Blight, and Healthy leaves.</p>
    </div>
    """, unsafe_allow_html=True)
