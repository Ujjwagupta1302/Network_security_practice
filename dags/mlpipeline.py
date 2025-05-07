from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import sys
import mlflow
import numpy as np
import pandas as pd
import pymongo
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, r2_score
from dotenv import load_dotenv
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier)
from urllib.parse import urlparse
from sklearn.model_selection import GridSearchCV
from network_security_practice.exception.exception import NetworkSecurityException
from network_security_practice.logging.logger import logging

# Load environment variables
load_dotenv()

mongo_db_url = os.getenv("MONGO_DB_URL")
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")

os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password


class NetworkModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def predict(self, x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise NetworkSecurityException(e, sys)


def export_data_from_mongodb():
    MONGO_DB_URL = mongo_db_url
    database_name = "UJJWALAI"  # Replace with actual name
    collection_name = "NetworkData"  # Replace with actual name
    mongo_client = pymongo.MongoClient(MONGO_DB_URL)
    collection = mongo_client[database_name][collection_name]
    df = pd.DataFrame(list(collection.find()))
    if "_id" in df.columns:
        df.drop(columns=["_id"], inplace=True)
    df.replace({"na": np.nan}, inplace=True)
    df.to_csv('phisingData.csv', index=False)
    logging.info("Data exported from MongoDB to phisingData.csv")


def split_data():
    dataframe = pd.read_csv('phisingData.csv')
    train_set, test_set = train_test_split(dataframe, test_size=0.2, random_state=42)
    train_set.to_csv('train.csv', index=False)
    test_set.to_csv('test.csv', index=False)
    logging.info("Data split into train.csv and test.csv")


def validate_data():
    pd.read_csv('train.csv').to_csv('train_validated.csv', index=False)
    pd.read_csv('test.csv').to_csv('test_validated.csv', index=False)
    logging.info("Validation complete: train_validated.csv and test_validated.csv generated.")


def transform_data():
    train_df = pd.read_csv("train_validated.csv")
    test_df = pd.read_csv("test_validated.csv")
    x_train = train_df.drop(columns=["Result"])
    y_train = train_df["Result"].replace(-1, 0)
    x_test = test_df.drop(columns=["Result"])
    y_test = test_df["Result"].replace(-1, 0)

    imputer = KNNImputer(n_neighbors=3)
    preprocessor = Pipeline([("imputer", imputer)])
    x_train_trans = preprocessor.fit_transform(x_train)
    x_test_trans = preprocessor.transform(x_test)

    np.save("train_array.npy", np.c_[x_train_trans, y_train])
    np.save("test_array.npy", np.c_[x_test_trans, y_test])

    with open("preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    logging.info("Data transformed and preprocessor saved.")


def train_model():
    train_arr = np.load("train_array.npy")
    test_arr = np.load("test_array.npy")
    x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
    x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

    models = {
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "AdaBoost": AdaBoostClassifier()
    }

    params = {
        "Decision Tree": {'criterion': ['gini', 'entropy']},
        "Random Forest": {'n_estimators': [16, 32, 64]},
        "Gradient Boosting": {'learning_rate': [0.1, 0.01], 'n_estimators': [64, 128]},
        "Logistic Regression": {},
        "AdaBoost": {'learning_rate': [0.1, 0.01], 'n_estimators': [64, 128]}
    }

    model_report = {}
    best_model = None
    best_model_name = None
    best_score = -np.inf

    for name, model in models.items():
        grid = GridSearchCV(model, params[name], cv=3, scoring='r2', error_score='raise')
        grid.fit(x_train, y_train)
        candidate = grid.best_estimator_
        score = r2_score(y_test, candidate.predict(x_test))
        model_report[name] = score

        if score > best_score:
            best_score = score
            best_model_name = name
            best_model = candidate

    # Log to MLflow
    f1 = f1_score(y_train, best_model.predict(x_train))
    precision = precision_score(y_train, best_model.predict(x_train))
    recall = recall_score(y_train, best_model.predict(x_train))

    mlflow.set_tracking_uri(mlflow_uri)
    with mlflow.start_run():
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall_score", recall)
        mlflow.log_metric("r2_score", best_score)

        if urlparse(mlflow_uri).scheme != "file":
            mlflow.sklearn.log_model(best_model, "model", registered_model_name=best_model_name)
        else:
            mlflow.sklearn.log_model(best_model, "model")

    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open("model.pkl", "wb") as f:
        pickle.dump(NetworkModel(preprocessor, best_model), f)

    logging.info(f"Best model ({best_model_name}) trained and saved as model.pkl")


# Define DAG
with DAG(
    dag_id='network_security_pipeline',
    default_args={
        'owner': 'airflow',
        'depends_on_past': False,
        'start_date': datetime(2023, 1, 1),
        'retries': 1,
    },
    schedule='@daily',
    catchup=False
) as dag:
    export_data = PythonOperator(
        task_id='export_data_from_mongodb',
        python_callable=export_data_from_mongodb
    )

    split_data_task = PythonOperator(
        task_id='split_data',
        python_callable=split_data
    )

    validate_data_task = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data
    )

    transform_data_task = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data
    )

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )

    # Task chaining
    export_data >> split_data_task >> validate_data_task >> transform_data_task >> train_model_task
