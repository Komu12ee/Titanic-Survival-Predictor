import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        Pclass:int,
        Sex: str,
        Age: int,
        SibSp: int,
        Parch: int,
        Fare: int,
        Embarked: str):

        self.sex = Sex

        self.embarked = Embarked

        self.pclass = Pclass

        self.age = Age

        self.sibsp = SibSp

        self.fare = Fare

        self.parch = Parch

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Pclass": [self.pclass],
                "Sex": [self.sex],
                "Age": [self.age],
                "SibSp": [self.sibsp],
                "Parch": [self.parch],
                "Fare": [self.fare],
                "Embarked": [self.embarked],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

