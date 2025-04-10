# Transformer Condition Predictor

## Overview
The **Transformer Condition Predictor** is a machine learning-based application designed to assess the health of electrical transformers by predicting their condition—**Normal**, **Not Good**, or **Severe**—based on key operational parameters. This project leverages a Multi-Layer Perceptron (MLP) model trained on synthetic transformer data and is deployed via a user-friendly Streamlit web app. It supports two input methods: CSV file upload and manual parameter entry.

This tool is part of my Final Year Project (FYP) in Electrical Engineering, aimed at enhancing predictive maintenance for power systems using AI.

## Features
- **Input Options**:
  - **CSV Upload**: Upload a dataset and predict the condition of a randomly selected transformer.
  - **Manual Input**: Enter parameters like Temperature, Oil Level, Current, Voltage, and Vibration manually.
- **Normalization**: Input data is normalized using a pre-fitted `MinMaxScaler` to match the model’s training scale.
- **Prediction**: Outputs the transformer’s condition with color-coded results:
  - Green: Normal
  - Yellow: Not Good
  - Red: Severe
- **Model**: Pre-trained MLP saved as `mlp_model.keras`.

## Prerequisites
- **Python 3.11+**
- **Dependencies**:
  ```bash
  pip install streamlit pandas numpy tensorflow scikit-learn joblib
