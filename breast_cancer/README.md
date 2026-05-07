# Breast Cancer Prediction using SVM
# Project Overview
This project demonstrates how to predict whether a breast tumor is malignant or benign using Support Vector Machine (SVM).
SVM is a supervised learning algorithm that finds the optimal decision boundary to separate classes, making it effective for medical diagnosis tasks.

# Dataset
**Source**: Breast Cancer Wisconsin (Diagnostic) Dataset (available in sklearn.datasets)//Breast_cancer_dataset.csv

**Samples**: 569

**Features**:
30 numerical attributes describing cell nucleus characteristics (mean, standard error, and worst values of radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, fractal dimension)

**features_used**:radius_mean, perimeter_mean, area_mean, compactness_mean, concavity_mean, concave_points_mean, radius_se, area_se, radius_worst, perimeter_worst, area_worst, compactness_worst, concavity_worst, concave_points_worst

**Target**:

0 → Malignant

1 → Benign

## Requirements
Install the dependencies before running the project:

# bash
pip install numpy pandas scikit-learn matplotlib

## Project Workflow
**Data Loading**: Import dataset from sklearn.datasets.

**Exploratory Data Analysis (EDA)**: Check class distribution and visualize feature correlations.

**Preprocessing**: Standardize features using StandardScaler and split dataset into training/testing sets.

**Model Training**: Train SVM classifier (sklearn.svm.SVC) with different kernels (linear, rbf, poly)//used rbf.

**Evaluation**: Measure accuracy, precision, recall, F1-score, and plot confusion matrix & ROC curve.

**Prediction**: Test model on unseen data and predict tumor type.