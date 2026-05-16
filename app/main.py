import streamlit as st
import requests
import subprocess
import time

#subprocess.Popen(["uvicorn","app.api:app","--reload"])
time.sleep(2)
st.title("Spam Detection App")
st.write("Entrez un message pour voir s'il s'agit d'un spam ou non.")

user_input = st.text_area("Message")
#-------------------------Prediction -----------------------------------------------------
if st.button("Predire"):
    try:
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
    except requests.exceptions.ConnectionError:
        st.error("L'API est eteint")
        


#---------------------------historique-----------------------------------------------------
st.subheader("message history")

history = requests.get(
    "http://127.0.0.1:8000/message"
)
message = history.json()['message']
for msg in message:
    st.write(f"""
            Message : {msg[1]}
            Prediction : {msg[2]}
            Date : {msg[3]}
""")
