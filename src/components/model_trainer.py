import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

#  ........MY EDIT.............
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
model = XGBClassifier(eval_metric='logloss')  # no "use_label_encoder"

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "XGBClassifier": XGBClassifier(eval_metric='logloss'),
                "CatBoost Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }
            params = {
                "Decision Tree": {
                  'criterion': ['gini', 'entropy', 'log_loss'],
                 # 'splitter': ['best', 'random'],
                 # 'max_features': ['sqrt', 'log2'],
                },
                "Random Forest": {
        # 'criterion': ['gini', 'entropy', 'log_loss'],
        # 'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        # 'criterion': ['friedman_mse'],
        # 'max_features': ['sqrt', 'log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Logistic Regression": {
                 'C': [0.1, 1, 10, 100],       # regularization strength
                 'solver': ["saga"],
                 'penalty': ['l1', 'l2', 'elasticnet', 'none']
                },
                "XGBClassifier": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [3, 5, 7]
                },
                "CatBoost Classifier": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Classifier": {
                   'learning_rate': [0.1, 0.01, 0.5, 0.001],
                   'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }


            

            model_report: dict = evaluate_models(
                 X_train=X_train, y_train=y_train,
                 X_test=X_test, y_test=y_test,
                 models=models, param=params
                )

# pick best model based on Accuracy
            best_model_name = None
            best_model_score = 0

            for name, metrics in model_report.items():
              if metrics["Accuracy"] > best_model_score:
                best_model_score = metrics["Accuracy"]
                best_model_name = name

            best_model = models[best_model_name]

            if best_model_score < 0.6:
              raise CustomException("No best model found")

            logging.info(f"Best found model: {best_model_name} with Accuracy: {best_model_score:.4f}")

            save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
            )

# Final evaluation on test set
            predicted = best_model.predict(X_test)
            final_accuracy = accuracy_score(y_test, predicted)

            return final_accuracy



            
        except Exception as e:
            raise CustomException(e,sys)