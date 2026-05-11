import os
import pandas as pd 
import numpy as np

class preprocessing():
    def __init__(self,file_path):
        self.data_path = file_path

    def data_load(self):
        self.data = pd.read_csv(self.data_path,usecols=['location', 'area', 'bedrooms', 'bathrooms', 'crime_rate', 'income_level', 'price'])
        return self.data

    def data_type(self):
        self.data['location'] = self.data['location'].map({'low': 0, 'medium': 1, 'premium': 2})
        self.data['income_level'] = self.data['income_level'].map({'low': 0, 'mid': 1, 'high': 2})
        self.data[['crime_rate', 'price']] = self.data[['crime_rate', 'price']].astype(np.float32)
        self.data[['location', 'income_level']] = self.data[['location', 'income_level']].astype(np.int8)
        other_col = self.data.columns.difference(['location', 'income_level', 'crime_rate', 'price'])
        self.data[other_col] = self.data[other_col].astype(np.int32)
        return self.data
        
    def feature_construction(self):
        self.data["rooms_total"] = self.data["bedrooms"] + self.data["bathrooms"]
        self.data["area_per_room"] = self.data["area"] / self.data["rooms_total"]
        return self.data["area_per_room"], self.data["rooms_total"]

    
def main():
    file_path = os.path.join(os.path.dirname(__file__),'../data/house_price_50k.csv')
    preprocess = preprocessing(file_path)
    preprocess.data_load()
    preprocess.data_type()
    preprocess.feature_construction()
    return preprocess.data

if __name__ == "__main__":
    print(main().head())


