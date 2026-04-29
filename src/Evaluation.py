import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

data = pd.read_csv("Dataset/spam.csv")

x = data['Message']
y = data['Category']

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size= 0.2,random_state=42)
model = joblib.load("model/spam_model.pkl")
y_pred = model.predict(x_test)
print("==============Classification report =====================")
print(classification_report(y_test,y_pred))
print("===============confusion Matrix===================")
print(confusion_matrix(y_test,y_pred))