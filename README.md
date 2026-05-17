# Spam Detector 🚀

This project is a machine learning model that detects whether a message is spam or not.

## 📌 Features

- Text preprocessing (cleaning, tokenization)
- Count vectorization
- Classification model (Naive Bayes)
- Evaluation with accuracy and classification report

## 📊 Results

              precision    recall  f1-score   support

         ham       0.99      0.99      0.99       966
        spam       0.95      0.94      0.95       149

    accuracy                           0.99      1115
   macro avg       0.97      0.97      0.97      1115
weighted avg       0.99      0.99      0.99      1115

## 🧠 Model

The model is trained using Scikit-learn with CountVectorizer features.

## 🚀 How to run

bash

pip install -r requirements.txt
python src/train.py

Or run the app:

bash
streamlit run app.py

## 📁 Project structure

spam-detector/
│
├── data/
|   ├── data-en-hi-de-fr.csv
|   ├── spam.csv
|
├── models/
|   ├── spam_model.pkl
├── notebooks/
|   ├── notebook.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│
├── app/
|   |── app.py
|   ├── api.py
|   ├── main.py
├── requirements.txt
├── README.md
├── message.db

## 👤 Author

Gloire Bisimwa

## ⭐ Future Improvements

- Try deep learning (LSTM / Transformers)
