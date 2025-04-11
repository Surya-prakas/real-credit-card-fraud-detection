from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

app = FastAPI()

# CORS to allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the new real model
model = joblib.load("real_fraud_model.pkl")

@app.get("/")
def home():
    return {"message": "Real Feature Fraud Detection API Running"}

@app.post("/predict")
async def predict(data: dict):
    # Extract the real features from user input
    features = np.array([
        data['amount'],
        data['day'],
        data['month'],
        data['hour'],
        data['cardLength'],
        data['isDaytime']
    ]).reshape(1, -1)

    prediction = model.predict(features)
    return {"fraud": bool(prediction[0])}
