# Spam Detector 🚀

This project is a machine learning model that detects whether a message is spam or not.

## 📌 Features
- Text preprocessing (cleaning, tokenization)
- TF-IDF vectorization
- Classification model (Naive Bayes)
- Evaluation with accuracy and classification report

## 📊 Results
- Accuracy: 0.99
- F1-score: 0.95

## 🧠 Model
The model is trained using Scikit-learn with TF-IDF features.

## 🚀 How to run

```bash
pip install -r requirements.txt
python src/train.py
```

Or run the app:

```bash
streamlit run app.py
```

## 📁 Project structure
spam-detector/
│
├── data/                
├── models/              
├── notebooks/           
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│
├── app/
|   |── main.py            
├── requirements.txt
├── README.md

## 👤 Author
Gloire Bisimwa