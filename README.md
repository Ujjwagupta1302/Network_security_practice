# 🚨 Phishing URL Detection with MLOps Automation
This project automates the **training, tracking, and deployment** of a phishing URL detection model using a full MLOps pipeline. It uses Apache Airflow (via Astronomer in Docker) for automation, MongoDB for data storage, DagsHub with MLflow for model tracking, and FastAPI to deploy the latest model. The system runs on a **daily schedule**, constantly improving itself with new data.

## 🔍 Project Highlights
- ✅ End-to-end MLOps pipeline with automated retraining
- ✅ Data pulled and processed from MongoDB
- ✅ Validation & model drift detection integrated into Airflow
- ✅ Hyperparameter tuning using GridSearchCV
- ✅ Best model versioned and logged to DagsHub with MLflow
- ✅ Latest model auto-deployed using FastAPI
- ✅ REST API for real-time phishing detection

## 🧠 Tech Stack

| Component               | Tool/Library                    |
|-------------------------|---------------------------------|
| Data Storage            | MongoDB                         |
| Pipeline Orchestration  | Apache Airflow (via Astronomer)|
| Containerization        | Docker                          |
| Model Training          | scikit-learn                    |
| Hyperparameter Tuning   | GridSearchCV                    |
| Model Tracking          | MLflow + DagsHub                |
| Deployment              | FastAPI                         |
| Missing Data Handling   | KNNImputer                      |

## 🧪 Model Training & Tuning
During training, we used GridSearchCV to select the best-performing model and hyperparameters from the following options:

**models** :  
    1. RandomForestClassifier  
    2. DecisionTreeClassifier  
    3. GradientBoostingClassifier  
    4. LogisticRegression(max_iter=1000)   
    5. AdaBoostClassifier  

## ✅ Preprocessing & Validation  

    1. Used KNNImputer to handle missing/null values.  
    2. Validated schema and content integrity in a dedicated Airflow validation step.  
    3. Added model drift detection logic.

## Running entire pipeline with Apache Airflow and Astronomer (Dockerized)
We use Astronomer CLI to run Airflow inside Docker for isolated, reproducible workflows.  
    1. Install the Astronomer CLI  
    2. Start the Airflow environment -> astro dev start  
    3. Access the Airflow web UI: -> http://localhost:8080/docs

## 🌐 Running FASTAPI Server
Command for loading :-  uvicorn api.main:app --reload  
Once started via above command model will be available at :- http://localhost:8000/predict  

## 🧠 How Entire Project Works
1. **Daily Airflow DAG**
    -> Pulls latest data from MongoDB  
    -> Validates the data and imputes missing values  
    -> Checks for model drift  
    -> Performs hyperparameter tuning via GridSearchCV  
    -> Logs best model to DagsHub with MLflow  
    -> Updates deployed model via FastAPI

2. **FastAPI Server**
    -> Loads the latest model  
    -> Exposes a /predict API to detect phishing URLs  
