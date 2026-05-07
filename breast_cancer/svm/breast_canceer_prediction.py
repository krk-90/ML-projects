import os
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

class BreastCancerPrediction:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = SVC(kernel='rbf', gamma='scale', random_state=42)

    def load_data(self, file_path): 
        # Load the dataset  
        self.data = pd.read_csv(file_path) 
        return self.data

    def clean_and_preprocess(self):
        # Drop the 'id' column
        self.data = self.data.drop(['id'], axis=1)
        
        # Map the 'diagnosis' column
        self.data['diagnosis'] = self.data['diagnosis'].map({'M': 1, 'B': 0})
        
        # Convert data types
        self.data = self.data.astype(np.float16)
        self.data['diagnosis'] = self.data['diagnosis'].astype(np.int8)
        
        # Drop columns with low correlation
        for col in self.data.corr().columns:
            if abs(self.data.corr().loc[col, 'diagnosis']) < 0.5:
                self.data = self.data.drop([col], axis=1)
        
        return self.data

    def prepare_data(self):
        # Prepare the data for training
        self.x = self.data.drop(['diagnosis'], axis=1)
        self.y = self.data['diagnosis']
        return self.x
    
    def split_data(self, test_size=0.2, random_state=42):
        # Split the data into training and testing sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=test_size, random_state=random_state)
        return self.x_train, self.x_test, self.y_train, self.y_test

    def process_data(self):
        #feature scaling
        self.x_train_scaled = self.scaler.fit_transform(self.x_train)
        self.x_test_scaled = self.scaler.transform(self.x_test)
        return self.x_train_scaled, self.x_test_scaled
    
    def train_model(self):
        #training the model
        self.model.fit(self.x_train_scaled, self.y_train)
        return self.model

    def evaluate_model(self):
        # evaluating the model
        self.y_pred = self.model.predict(self.x_test_scaled)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.classification_report = classification_report(self.y_test, self.y_pred)
        self.confusion_matrix = confusion_matrix(self.y_test, self.y_pred)
        return self.accuracy, self.classification_report, self.confusion_matrix
    
    
    def predict_new(self,value):
        # predicting a new case
        self.y_pred = self.model.predict(value)
        return self.y_pred
    

def main():
    try:
        file_path = os.path.join(os.path.dirname(__file__), '../data/Breast_cancer_dataset.csv')

        prediction = BreastCancerPrediction()
        data = prediction.load_data(file_path)
        data = prediction.clean_and_preprocess()
        prediction.prepare_data()
        prediction.split_data()
        prediction.process_data()
        prediction.train_model()
        evaluation = prediction.evaluate_model()
        print(f"Accuracy: {evaluation[0]}")
        print(f"Classification Report:\n{evaluation[1]}")
        print(f"Confusion Matrix:\n{evaluation[2]}")
        while True:
            user_input = input("\nDo you want to predict a new case? (y/n): ")
            if user_input.lower() != 'y':
                break
            else:
                try:
                    radius_mean   = float(input("Enter radius_mean: "))
                    perimeter_mean  = float(input("Enter perimeter_mean: "))
                    area_mean    = float(input("Enter area_mean: "))
                    compactness_mean = float(input("Enter compactness_mean: "))
                    concavity_mean      = float(input("Enter concavity_mean: "))
                    concave_points_mean = float(input("Enter concave_points_mean: "))
                    radius_se   = float(input("Enter radius_se: "))
                    area_se      = float(input("Enter area_se: "))
                    radius_worst      = float(input("Enter radius_worst: "))
                    perimeter_worst  = float(input("Enter perimeter_worst: "))
                    area_worst       = float(input("Enter area_worst: "))
                    compactness_worst    = float(input("Enter compactness_worst: "))
                    concavity_worst    = float(input("Enter concavity_worst: "))
                    concave_points_worst    = float(input("Enter concave_points_worst: "))

                    data_input = [radius_mean, perimeter_mean, area_mean, compactness_mean, concavity_mean, concave_points_mean, radius_se, area_se, radius_worst, perimeter_worst, area_worst, compactness_worst, concavity_worst, concave_points_worst]
                    feature_values = data_input.split(',')
                    predicted_class = prediction.predict_new(np.array(feature_values).reshape(1, -1))
                    print(f"Predicted Class: {'Malignant' if predicted_class[0] == 1 else 'Benign'}")

                except ValueError:
                    print("Invalid input. Please enter numeric values separated by commas.")

    except Exception as e:
        print(f"An error occurred: {e}")    

if __name__ == "__main__":
    main()