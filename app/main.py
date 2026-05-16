import streamlit as st
import requests
st.title("Spam Detection App")
st.write("Entrez un message pour voir s'il s'agit d'un spam ou non.")

user_input = st.text_area("Message")

if st.button("Predire"):
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"text":user_input}
    )
    result = response.json()
    prediction = result["prediction"]
    if prediction == "spam":
        st.error("Spam détecter")
    else:
        st.success("Not spam")