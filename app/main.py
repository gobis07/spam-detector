import streamlit as st
import joblib

model = joblib.load("model/spam_model.pkl")
st.title("Spam Detection App")
st.write("Entrez un message pour voir s'il s'agit d'un spam ou non.")

user_input = st.text_area("Message")

if st.button("Predire"):
    if user_input.strip() != "":
        prediction = model.predict([user_input])[0]
        if prediction == 'spam':
            st.error("Spam detecté")
        else:
            st.success("Message Normal (Ham)")
    else:
        st.warning("Veuillez entrer un message")