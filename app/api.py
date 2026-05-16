from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model/spam_model.pkl")

class Message(BaseModel):
    text:str

@app.post("/predict")
def predict(data:Message):

    prediction = model.predict([data.text])

    return{
        "prediction":prediction[0]
    }