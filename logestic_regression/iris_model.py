import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import os

class iris_model:
    def __init__(self,data):
        self.data = data.drop(columns = ['Id'])

    def encode_label(self):
        self.le = preprocessing.LabelEncoder()
        self.data['Species']= self.le.fit_transform(self.data['Species'])
        self.x = self.data.drop('Species',axis = 1)
        self.y = self.data['Species']
        return self.data['Species'],self.x,self.y
    
    def train_test(self):
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.x,self.y,test_size=0.2,random_state=2)
        return  self.x_train,self.x_test,self.y_train,self.y_test
    
    def scaler_data(self):
        self.scaler = StandardScaler()
        self.x_train_scaler = self.scaler.fit_transform(self.x_train)
        self.x_test_scaler = self.scaler.transform(self.x_test)
        return  self.x_train_scaler,self.x_test_scaler
    
    def model(self):
        self.log_model = LogisticRegression(max_iter=200)
        return self.log_model

    def train(self):
        self.train_model = self.log_model.fit(self.x_train_scaler,self.y_train)
        return self.train_model

    def predict(self):
        self.prediction = self.train_model.predict(self.x_test_scaler)
        return self.prediction

    def decode_label(self):
        self.prediction_decode = self.le.inverse_transform(self.prediction)
        return self.prediction_decode

    def accuracy(self):
        self.score = accuracy_score(self.y_test,self.prediction)
        return self.score

    def predict_new(self, flower_features):
    # flower_features = [[5.1, 3.5, 1.4, 0.2]]
        self.scaled = self.scaler.transform(flower_features)
        self.predicted = self.train_model.predict(self.scaled)
        return self.le.inverse_transform(self.predicted)

def main():  
    try:
        data_path = os.path.join(os.path.dirname(__file__), '../data/Iris.csv')
        data = pd.read_csv(data_path)
        model = iris_model(data)
        model.encode_label()
        model.train_test()
        model.scaler_data()
        model.model()
        model.train()
        model.predict()
        prediction = model.decode_label().tolist()
        #print(prediction)
 

        sepal_length = float(input("Enter sepal length: "))
        sepal_width = float(input("Enter sepal width: "))
        petal_length = float(input("Enter petal length: "))
        petal_width = float(input("Enter petal width: "))
        
        feature = [[sepal_length, sepal_width, petal_length, petal_width]]
        predicted_species = model.predict_new(feature)
        print(f"\nThe predicted species is: {predicted_species[0]}")
        print(f"Model Training Accuracy: {model.accuracy()}")

    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
    except ValueError:
        print("\nError: Please enter numbers only for the flower features.")


if __name__ == "__main__":
    main()