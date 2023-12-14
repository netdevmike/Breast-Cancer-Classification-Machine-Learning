# Breast Cancer Diagnostic Model

This project includes a machine learning model for diagnosing breast cancer using the Breast Cancer Wisconsin (Diagnostic) Data Set. The model is built using a deep neural network implemented with Keras and evaluated using K-Fold Cross-Validation to ensure reliability and consistency.

## Dataset

The dataset used is the Breast Cancer Wisconsin (Diagnostic) Data Set from the UCI Machine Learning Repository. It includes features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass, describing characteristics of the cell nuclei present in the image.

## Features

The project includes:
- Data preprocessing with standard scaling.
- A deep neural network model with dropout regularization.
- K-Fold Cross-Validation for model evaluation.
- Early stopping to prevent overfitting.
- Plotting of accuracy and loss for both training and validation sets.

## Requirements

To run this code, you will need the following libraries:
- Numpy
- Matplotlib
- Scikit-learn
- Keras

You can install these with pip using the following command:

```bash
pip install numpy matplotlib scikit-learn keras
