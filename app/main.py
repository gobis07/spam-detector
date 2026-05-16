import streamlit as st
import requests
import time
import subprocess

subprocess.Popen(["uvicorn","app.api:app","--reload"])
time.sleep(2)
st.set_page_config(
    page_title="Spam Detection App",
    page_icon="🛡️",
    layout="centered"
)
# Petit CSS pour faire des cartes
st.markdown("""
<style>
    .spam-card {
        background-color: #2b2b2b;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #ff4b4b;
    }
    .ham-card {
        background-color: #2b2b2b;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #00c853;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)
st.title("🛡️ Spam Detection App")
st.caption("Entrez un message pour voir s'il s'agit d'un spam ou non")
# Zone de saisie + bouton sur la même ligne
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_area("Message", placeholder="Tape ton SMS ou email ici...", height=120, label_visibility="collapsed")
with col2:
    st.write("")
    st.write("")
    predict_btn = st.button("Predire", type="primary")
# Prédiction
if predict_btn and user_input:
    with st.spinner("Analyse en cours..."):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"text": user_input},
                timeout=5
            )
            result = response.json()
            prediction = result["prediction"]
           
            if prediction == "spam":
                st.error("🚨 Spam détecté")
            else:
                st.success("✅ Message légitime")
           
            time.sleep(0.5)
            st.rerun()
           
        except requests.exceptions.ConnectionError:
            st.error("L'API est éteinte. Lance `uvicorn app.api:app --reload` d'abord.")
st.divider()
st.subheader("📜 Historique des messages")
# Historique en cartes
try:
    history = requests.get("http://127.0.0.1:8000/message").json()["message"]
   
    if not history:
        st.info("Aucun message pour l'instant")
    else:
        for msg in reversed(history):  # Plus récent en haut
            msg_id, text, pred, date = msg
            css_class = "spam-card" if pred == "spam" else "ham-card"
            icon = "🚨" if pred == "spam" else "✅"
           
            st.markdown(f"""
            <div class="{css_class}">
                <strong>{icon} {pred.upper()}</strong> • {date}<br>
                <span style="opacity: 0.8;">{text}</span>
            </div>
            """, unsafe_allow_html=True)
           
except Exception as e:
    st.warning("Impossible de charger l'historique")