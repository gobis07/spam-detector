# Spam Detector 🚀

C'est un projet de machine learning qui détecte si un message est un spam ou non.
J'utilise scikit-learning pour l'entrainement et la vectorization du texte,streamlite pour le frontend,
fastAPI pour le backend et sqlite3 pour la base de donnée

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

pour l'entrainement j'utilse MultinomialNB de la bibliothèque scikit-learn 
et pour la vectorization j'utilise CountVectorizer aussi de scikit-learn

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
