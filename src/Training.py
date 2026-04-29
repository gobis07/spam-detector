import pandas as pd
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from preprocess import get_vectorizer
from sklearn.pipeline import Pipeline

data = pd.read_csv("Dataset/spam.csv")

x = data['Message']
y = data['Category']

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size= 0.2,random_state=42)

model = Pipeline([
    ("vectorizer",get_vectorizer()),
    ("MNB",MultinomialNB())
])

model.fit(x_train,y_train)
joblib.dump(model, "model/spam_model.pkl")
print("Ok!!!!!")
