# Potato Disease Classifier
This repository contains a FastAPI backend and a Streamlit frontend for classifying potato plant diseases using deep learning.


![image](https://github.com/user-attachments/assets/ba76f5f3-2cc5-4c1b-801b-f39c625e199a)


## Overview
The project consists of two main components:

1. FastAPI Backend (api/main.py):

- Uses TensorFlow and Keras to load a pre-trained convolutional neural network (CNN) model (potatoes.h5).
- Exposes a /predict endpoint for classifying diseases based on uploaded potato plant leaf images.
- Provides a /ping endpoint for health checks.

2. Streamlit Frontend (app.py):

- Offers a user-friendly interface for uploading potato plant leaf images.
- Sends images to the FastAPI backend for disease classification.
- Displays classification results including disease type and confidence level.

## Getting Started
To run the application locally:

1. Clone the repository:

```bash
git clone https://github.com/TruongSinhAI/PotatoDiseaseDetector.git
cd PotatoDiseaseDetector
```

2. Install dependencies:

- Ensure Python 3.x is installed.
- Install required Python packages:
```bash
pip install -r requirements.txt
```
3. Run FastAPI Backend:

- Navigate to the api directory:
```bash
cd api
```
- Start the FastAPI server:
```bash
uvicorn main:app --reload
```
The FastAPI server will start at http://localhost:8080 by default.

4. Run Streamlit Frontend:

- In a new terminal window/tab, navigate to the root directory of the repository:
```bash
cd ..
```
- Start the Streamlit app:
```bash
streamlit run app.py
```
The Streamlit app will open in your default web browser.
5. Upload an image:

- Use the Streamlit interface to upload an image of a potato plant leaf.
- Click on "Classify Disease" to see the prediction results displayed.
## Model Details
The deep learning model used for disease classification is based on TensorFlow and Keras. It is trained on the [Plant Village](https://www.kaggle.com/datasets/arjuntejaswi/plant-village) dataset from Kaggle, which includes images of potato plant leaves affected by various diseases.
