import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler


class HousePricePredictor:
    def __init__(self):
        self.model = Ridge(alpha=1.0)
        self.scaler = StandardScaler()


    def load_data(self, file_path):
        self.data = pd.read_csv(file_path, usecols=['area', 'price'])
        return self.data
    
    def prepare_data(self):
        self.x = self.data['area'].values.reshape(-1, 1)
        self.y = self.data['price'].values
        return self.x,self.y
    

    def split_data(self,test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=test_size, random_state=random_state)
        return self.x_train, self.x_test, self.y_train, self.y_test
    def preprocess_data(self):
        self.x_train_scaled = self.scaler.fit_transform(self.x_train)
        self.x_test_scaled = self.scaler.transform(self.x_test)
        return self.x_train_scaled, self.x_test_scaled
    
    def train_model(self):
        self.model.fit(self.x_train_scaled, self.y_train)
        return self.model

    def predict(self):
        self.prediction = self.model.predict(self.x_test_scaled)
        return self.prediction

    def evaluate_model(self):
        self.mse = mean_squared_error(self.y_test, self.prediction)
        self.r2 = r2_score(self.y_test, self.prediction)
        self.mae = mean_absolute_error(self.y_test, self.prediction)
        return self.mse, self.r2, self.mae
    
    def predict_new(self, area):
        self.area_scaled = self.scaler.transform([[area]])
        self.predicted_price = self.model.predict(self.area_scaled)
        return self.predicted_price


def main():
    try:
        data_path = os.path.join(os.path.dirname(__file__), '../data/house_price_50k.csv')
        predictor = HousePricePredictor()
        predictor.load_data(data_path)
        predictor.prepare_data()
        predictor.split_data()
        predictor.preprocess_data()
        predictor.train_model()
        predictor.predict()
        mse, r2, mae = predictor.evaluate_model()
        print(f'root Mean Squared Error: {np.sqrt(mse)}')
        print(f'R^2 Score: {r2}')
        print(f'Mean Absolute Error: {mae}')
        print("Coefficients:", predictor.model.coef_)
        print("Intercept:", predictor.model.intercept_)

        while True:
            input_area = float(input("Enter the area of the house to predict its price: "))
            print(f'Predicted price for new area: {predictor.predict_new(input_area)}')
            if input("Do you want to predict another price? (yes/no): ").lower() != 'yes':
                break
    except Exception as e:
        print(f'An error occurred: {e}')
if __name__ == "__main__":        
    main()
