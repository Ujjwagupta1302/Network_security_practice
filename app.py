import mlflow
import os
from mlflow import pyfunc
from fastapi import FastAPI, Query
from pydantic import BaseModel
import logging
import numpy as np
import re
import socket
import ssl
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
# Set MLflow tracking URI
mlflow.set_tracking_uri(mlflow_uri)

FEATURE_ORDER = [
    "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol", 
    "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State", 
    "Domain_registeration_length", "Favicon", "port", "HTTPS_token", "Request_URL", 
    "URL_of_Anchor", "Links_in_tags", "SFH", "Submitting_to_email", "Abnormal_URL", 
    "Redirect", "on_mouseover", "RightClick", "popUpWidnow", "Iframe", "age_of_domain", 
    "DNSRecord", "web_traffic", "Page_Rank", "Google_Index", "Links_pointing_to_page", 
    "Statistical_report"
]

# Load model from MLflow
def get_latest_model():
    try:
        client = mlflow.tracking.MlflowClient()



        #latest_model_version = client.get_latest_versions("Random Forest", stages=["None"])[0].version
        
        # Fetch the latest version of the model from the Production stage
        latest_version = client.get_latest_versions(name = "Random Forest")[0].version
        client.transition_model_version_stage(
            name="Random Forest", 
            version=latest_version, 
            stage="Production"
        )
        
        # Transition the model to the Production stage if it's not already there
        model_uri = f"models:/Random Forest/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)
        return model,latest_version
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

# Feature extraction logic
def extract_features(url):
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    features = {}

    # Computed features
    features["having_IP_Address"] = 1 if re.match(r"\d+\.\d+\.\d+\.\d+", hostname) else 0
    features["URL_Length"] = len(url)
    features["Shortining_Service"] = 1 if any(short in url for short in ["bit.ly", "tinyurl", "goo.gl"]) else 0
    features["having_At_Symbol"] = 1 if "@" in url else 0
    features["double_slash_redirecting"] = 1 if url.count("//") > 1 else 0
    features["Prefix_Suffix"] = 1 if "-" in hostname else 0
    features["having_Sub_Domain"] = 1 if hostname.count('.') > 1 else 0

    try:
        cert = ssl.get_server_certificate((hostname, 443))
        features["SSLfinal_State"] = 1 if cert else 0
    except:
        features["SSLfinal_State"] = 0

    # Hardcoded features (adjust if you can extract later)
    features.update({
        "Domain_registeration_length": 1,
        "Favicon": 1,
        "port": 0,
        "HTTPS_token": 1 if "https" in url else 0,
        "Request_URL": 1,
        "URL_of_Anchor": 1,
        "Links_in_tags": 1,
        "SFH": 1,
        "Submitting_to_email": 0,
        "Abnormal_URL": 0,
        "Redirect": 0,
        "on_mouseover": 0,
        "RightClick": 0,
        "popUpWidnow": 0,
        "Iframe": 0,
        "age_of_domain": 1,
        "DNSRecord": 1,
        "web_traffic": 1,
        "Page_Rank": 1,
        "Google_Index": 1,
        "Links_pointing_to_page": 1,
        "Statistical_report": 0,
    })

    return features

# Convert to NumPy array in correct order
def prepare_features_array(feature_dict):
    return np.array([feature_dict[feature] for feature in FEATURE_ORDER]).reshape(1, -1)

# Prediction endpoint
@app.get("/predict")
async def predict(url: str = Query(..., description="Enter a full URL")):
    model,latest_version = get_latest_model()
    if not model:
        return {"error": "Model could not be loaded from MLflow"}

    try:
        features = extract_features(url)
        feature_array = prepare_features_array(features)
        prediction = model.predict(feature_array)
        output = prediction.tolist() 
        output.append(f"The latest version of the model used is {latest_version}")
        #print(latest_version)
        return {"prediction": output}
        
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return {"error": "Prediction process failed"}
