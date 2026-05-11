import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from joblib import dump, load


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from data_processing.preprocess import main

class random_forest_model():
    def __init__(self,dataset):
        self.data = dataset

#train_test function splits the dataset into training and testing sets. 
    def train_test(self):
        self.x = main().drop(labels = ['price'],axis=1)
        self.y = main()['price']
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.x,self.y,test_size=0.3,random_state=42)
        return self.x_train,self.x_test,self.y_train,self.y_test
    
#this function uses a pipeline to unify the preprocessing and modeling steps. 
    def model_pipeline(self): 
        self.pipe  = Pipeline([("random_forest_regressor", RandomForestRegressor(
            n_estimators=100, 
            max_features ="log2",
            max_depth=20, 
            min_samples_split=5, 
            min_samples_leaf=1, 
            random_state=42
        ))]) 
        return self.pipe
    
#train_model function fits the pipeline to the training data.
    def train_model(self):
        return self.pipe.fit(self.x_train,self.y_train)
    
#evaluate_model function uses the trained model to make predictions on the test set and returns the predicted values and r2 score, mean squared error, and mean absolute error as evaluation metrics.
    def evaluate_model(self):
        self.y_pred = self.pipe.predict(self.x_test)
        self.r2 = r2_score(self.y_test,self.y_pred)
        self.mse = mean_squared_error(self.y_test,self.y_pred)
        self.mae = mean_absolute_error(self.y_test,self.y_pred)
        return self.y_pred,self.r2, self.mse, self.mae
    
#feature_importance function retrieves the feature importance from the trained Random Forest model and returns it. 
    def feature_importance(self):
        self.feature_importance = self.pipe.named_steps['random_forest_regressor'].feature_importances_
        return self.feature_importance

    def new_prediction(self,new_data):
        self.new_pred = self.pipe.predict(new_data)
        return self.new_pred
    
#we can use this function to perform hyperparameter tuning using RandomizedSearchCV. It will search for the best combination of hyperparameters and update the model accordingly.
    """def hyperparameter_tuning(self):
        param_dist = {"n_estimators": [100, 200, 300],
                        "max_features": [1.0, "sqrt", "log2", None],
                        "max_depth": [None, 10, 20, 30],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4]}

        self.random_search = RandomizedSearchCV(estimator=self.pipe.named_steps["random_forest_regressor"],
                                                param_distributions=param_dist,
                                                n_iter=10,
                                                cv=5,
                                                verbose=2,
                                                random_state=42,
                                                n_jobs=1)
    
        self.random_search.fit(self.x_train, self.y_train)
        print(f"Best parameters: {self.random_search.best_params_}")
        print(f"Best score: {self.random_search.best_score_}")
        self.pipe.named_steps['random_forest_regressor'].set_params(**self.random_search.best_params_)
        self.train_model()"""

def model():
    try:
        file = main()
        random_forest = random_forest_model(file)
        random_forest.train_test()
        random_forest.model_pipeline()
        random_forest.train_model()
        score = random_forest.evaluate_model()
        random_forest.feature_importance()
        """random_forest.hyperparameter_tuning()
        random_forest.evaluate_model()"""
        print(f"R2 Score: {score[1]}")
        print(f"Root Mean Squared Error: {np.sqrt(score[2])}")
        print(f"Mean Absolute Error: {score[3]}")
        while True:
            user_input = input("\nDo you want to predict a new case? (y/n): ")
            if user_input.lower() != 'y':
                break
            else:
                area = float(input("Enter area: "))
                bedrooms = int(input("Enter number of bedrooms: "))
                bathrooms = int(input("Enter number of bathrooms: "))
                crime_rate = float(input("Enter crime rate: "))
                location = int(input("Enter location (0 for low, 1 for medium, 2 for premium): "))
                income_level = int(input("Enter income level: "))
                rooms_total = bedrooms + bathrooms
                area_per_room = area / rooms_total
                new_data = pd.DataFrame([{
                    "area": area,
                    "bedrooms": bedrooms,
                    "bathrooms": bathrooms,
                    "crime_rate": crime_rate,
                    "location": location,
                    "income_level": income_level,
                    "rooms_total": rooms_total,
                    "area_per_room": area_per_room
                }])
                prediction = random_forest.new_prediction(new_data)
                print(f"Predicted price: {prediction[0]}")
        return random_forest.pipe

    except Exception as e:
        print(f"An error occurred: {e}")        

if __name__ == "__main__":   
    trained_model = model()
    joblib.dump(trained_model,os.path.join(PROJECT_ROOT, 'random_forest_model', 'random_forest_model_v1.joblib'))
    joblib.load(os.path.join(PROJECT_ROOT, 'random_forest_model', 'random_forest_model_v1.joblib'))
