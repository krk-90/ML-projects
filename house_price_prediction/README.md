# House Price Prediction using Ridge Regression

## 📌 Project Overview
This project demonstrates how to predict house prices based on house area using **Ridge Regression**.
Ridge Regression is a regularized linear regression model that helps prevent overfitting by adding a 
penalty term to the cost function. This is ideal for real estate price prediction tasks.

---

## 📂 Dataset
- **Source**: `house_price_50k.csv` (50,000 samples)
- **Features**:
  - **Area** (in square feet): House area - the independent variable (X)
- **Target**:
  - **Price** (in currency): House price - the dependent variable (y)
- **Dataset Size**: 50,000 house records with area and price information

---

## ⚙️ Requirements
Install the dependencies before running the project:

```bash
pip install numpy pandas scikit-learn 