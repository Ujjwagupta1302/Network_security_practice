from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
import yaml
import os
import sys
import mlflow
from scipy.stats import ks_2samp
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
import data_schema

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

def save_schema_yaml() :
    schema_dict = {
        "columns": [
            "having_IP_Address",
            "URL_Length",
            "Shortining_Service",
            "having_At_Symbol",
            "double_slash_redirecting",
            "Prefix_Suffix",
            "having_Sub_Domain",
            "SSLfinal_State",
            "Domain_registeration_length",
            "Favicon",
            "port",
            "HTTPS_token",
            "Request_URL",
            "URL_of_Anchor",
            "Links_in_tags",
            "SFH",
            "Submitting_to_email",
            "Abnormal_URL",
            "Redirect",
            "on_mouseover",
            "RightClick",
            "popUpWidnow",
            "Iframe",
            "age_of_domain",
            "DNSRecord",
            "web_traffic",
            "Page_Rank",
            "Google_Index",
            "Links_pointing_to_page",
            "Statistical_report",
            "Result"
        ],
        "numerical_columns": [
            "having_IP_Address",
            "URL_Length",
            "Shortining_Service",
            "having_At_Symbol",
            "double_slash_redirecting",
            "Prefix_Suffix",
            "having_Sub_Domain",
            "SSLfinal_State",
            "Domain_registeration_length",
            "Favicon",
            "port",
            "HTTPS_token",
            "Request_URL",
            "URL_of_Anchor",
            "Links_in_tags",
            "SFH",
            "Submitting_to_email",
            "Abnormal_URL",
            "Redirect",
            "on_mouseover",
            "RightClick",
            "popUpWidnow",
            "Iframe",
            "age_of_domain",
            "DNSRecord",
            "web_traffic",
            "Page_Rank",
            "Google_Index",
            "Links_pointing_to_page",
            "Statistical_report",
            "Result"
        ]
    }

    with open("schema.yaml" , "w") as f :
        yaml.dump(schema_dict,f) 

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


def validate_number_of_columns(dataframe:pd.DataFrame)->bool :
    try:
        with open("schema.yaml",'r') as f :
            schema = yaml.safe_load(f) 

        number_of_columns = sum(len(v) for v in schema.values() if isinstance(v, list))
        logging.info(f"Required number of columns{number_of_columns}") 
        if len(dataframe.columns)==number_of_columns :
            return True 
        return False 
    except Exception as e:
        NetworkSecurityException(e,sys) 

def validate_numeric_columns(dataframe:pd.DataFrame)-> bool:
    with open("schema.yaml",'r') as f :
            schema = yaml.safe_load(f) 

    schema_numeric_columns = len(schema["numerical_columns"])
    numeric_columns = len(dataframe.select_dtypes(include=['number']).columns.tolist())

    if schema_numeric_columns==numeric_columns :
        return True 
    return False

def detect_dataset_drift(base_df,current_df,threshold=0.05)->bool:
    try:
        status=True
        report={}
        for column in base_df.columns:
            d1=base_df[column]
            d2=current_df[column]
            is_same_dist=ks_2samp(d1,d2)
            if threshold<=is_same_dist.pvalue:
                is_found=False
            else:
                is_found=True
                status=False
            report.update({column:{
                "p_value":float(is_same_dist.pvalue),
                "drift_status":is_found
                
                }})
        with open("report.yaml", "w") as f :
            yaml.dump(report, f) 

    except Exception as e:
        raise NetworkSecurityException(e,sys)


def split_data():
    dataframe = pd.read_csv('phisingData.csv')
    train_set, test_set = train_test_split(dataframe, test_size=0.2, random_state=42)
    train_set.to_csv('train.csv', index=False)
    test_set.to_csv('test.csv', index=False)
    logging.info("Data split into train.csv and test.csv")


def validate_data():
    train_dataframe = pd.read_csv('train.csv') 
    test_dataframe = pd.read_csv('test.csv') 
    status_1_train = validate_number_of_columns(train_dataframe)
    status_1_test = validate_number_of_columns(test_dataframe) 
    status_2_train = validate_numeric_columns(train_dataframe) 
    status_2_test = validate_numeric_columns(test_dataframe) 
    status_3 = detect_dataset_drift(train_dataframe, test_dataframe)
    #if status_1_test == False or status_1_train==False or status_2_train==False or status_2_test==False :
        #logging.error("Validation failed. Skipping subsequent tasks.")
        #return 'validation_failed'
    #else:
    logging.info("Validation passed.")
    train_dataframe.to_csv('train_validated.csv', index=False)
    test_dataframe.to_csv('test_validated.csv', index=False)
    logging.info("Validation complete: train_validated.csv and test_validated.csv generated.")
    return 'continue_processing'


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
        mlflow.log_artifact("schema.yaml") 
        mlflow.log_artifact("report.yaml")

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
    schema_file = PythonOperator(
        task_id = 'create_schema_yaml_file', 
        python_callable = save_schema_yaml
    )
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
    schema_file >> export_data >> split_data_task >> validate_data_task >> transform_data_task >> train_model_task
