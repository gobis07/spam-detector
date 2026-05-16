from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import sqlite3
app = FastAPI()
model = joblib.load("model/spam_model.pkl")

class Message(BaseModel):
    text:str

@app.post("/predict")
def predict(data:Message):

    prediction = model.predict([data.text])[0]
    conn = sqlite3.connect("message.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO message(text,prediction)
        VALUES (?,?)
        """,
        (data.text,prediction)
    )
    conn.commit()
    conn.close()
    return{
        "prediction":prediction
    }
@app.get("/message")
def get_messages():
    conn = sqlite3.connect("message.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM message")
    message = cursor.fetchall()
    conn.close()
    return{"message":message}