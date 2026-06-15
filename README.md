# Handwritten Digit Classifier

## Overview

This project is a Deep Learning-based Handwritten Digit Recognition System built using TensorFlow, Keras, and Streamlit. The model is trained on the MNIST dataset and can classify handwritten digits from 0 to 9 with high accuracy.

## Features

* Handwritten digit recognition
* CNN-based image classification
* Image upload support
* Real-time prediction
* Confidence score display

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Streamlit
* Pillow

## Dataset

MNIST Handwritten Digits Dataset

* 60,000 training images
* 10,000 testing images
* 10 classes (0–9)

## Model Performance

* Accuracy: 98.64%
* Epochs: 3

## Run the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

## Project Structure

```text
├── app.py
├── train.py
├── model.keras
├── requirements.txt
└── README.md
```
