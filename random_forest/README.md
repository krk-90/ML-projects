## House Price Prediction using RandomForestRegressor
## 📌Project Overview
This project builds a machine learning regression model to predict house prices based on key features such as location, area, bedrooms, bathrooms, crime rate, and income level.
We use RandomForestRegressor, an ensemble learning method that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

## Dataset
 **Source**: `house_price_50k.csv` (50,000 samples)
- **Features**:
  - **Area** (in square feet): House area - the independent variable (X)
- **Target**:
  - **Price** (in currency): House price - the dependent variable (y)
- **Dataset Size**: 50,000 house records with area and price information
## The dataset contains the following columns:

**location**: Categorical feature representing the neighborhood/region.

**area**: Numeric feature representing the size of the house (square feet).

*bedrooms*: Number of bedrooms.

**bathrooms**: Number of bathrooms.

**crime_rate**: Numeric feature representing crime rate in the area.

**income_level**: Categorical feature representing average income level of residents.

**rooms_total,area_per_room**:both are constructed feature.

**price**: Target variable (house price).

## We load only these columns using:

**usecols** = ['location', 'area', 'bedrooms', 'bathrooms', 'crime_rate', 'income_level', 'price']

## ⚙️ Requirements
Install the dependencies before running the project:

```bash
pip install numpy pandas scikit-learn