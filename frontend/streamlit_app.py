import streamlit as st
import requests

import os
API = os.getenv("API_URL", "http://127.0.0.1:8001")

st.set_page_config(page_title="AI Health Assistant (Educational)", layout="centered")

st.title("AI-driven Symptom-based Health Recommendation System")
st.caption("Educational tool only — not a doctor, not a diagnosis.")

# ---------- session state ----------
if "sid" not in st.session_state:
    st.session_state.sid = None
if "top" not in st.session_state:
    st.session_state.top = None
if "next_symptom" not in st.session_state:
    st.session_state.next_symptom = None
if "next_question" not in st.session_state:
    st.session_state.next_question = None
if "stop" not in st.session_state:
    st.session_state.stop = False

# ---------- start ----------
with st.container(border=True):
    st.subheader("1) Start")
    if st.session_state.sid is None:
        if st.button("Start session"):
            r = requests.post(f"{API}/start_session")
            data = r.json()
            st.session_state.sid = data["session_id"]
            st.success("Session started.")
            st.info(data["disclaimer"])
            st.rerun()
    else:
        st.success("Session active.")
        st.code(st.session_state.sid)

sid = st.session_state.sid
if sid is None:
    st.stop()

# ---------- initial symptoms ----------
COMMON_SYMPTOMS = [
    "fever","cough","fatigue","sore_throat","runny_nose","nasal_congestion",
    "shortness_of_breath","wheezing","chest_pain",
    "nausea","vomiting","diarrhea","abdominal_pain",
    "headache","severe_headache","dizziness","light_sensitivity",
    "burning_urination","frequent_urination","blood_in_urine","flank_pain",
    "rash","itching","hives",
    "anxiety","restlessness","sleep_disturbance",
    "loss_of_taste_smell"
]

with st.container(border=True):
    st.subheader("2) Select symptoms you have (3–8 is ideal)")
    selected = st.multiselect("Choose symptoms:", COMMON_SYMPTOMS)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Submit selected symptoms"):
            for s in selected:
                requests.post(f"{API}/answer", json={"session_id": sid, "symptom": s, "value": 1})
            st.success("Saved. Now click 'Get next question'.")
    with col2:
        if st.button("Get next question"):
            r = requests.get(f"{API}/next", params={"session_id": sid})
            data = r.json()
            st.session_state.top = list(zip(data["top_diseases"], data["top_probs"]))
            st.session_state.next_symptom = data.get("next_symptom")
            st.session_state.next_question = data.get("next_question")
            st.session_state.stop = data.get("stop", False)
            st.rerun()

# ---------- show top + next question ----------
if st.session_state.top:
    with st.container(border=True):
        st.subheader("3) Current top matches (educational)")
        for d, p in st.session_state.top:
            st.write(f"- **{d}**: {p:.3f}")

        if st.session_state.stop:
            st.success("Enough information collected. Show final output below.")
        else:
            st.subheader("Follow-up question")
            st.info(st.session_state.next_question)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Yes"):
                    requests.post(f"{API}/answer", json={"session_id": sid, "symptom": st.session_state.next_symptom, "value": 1})
                    st.session_state.next_symptom = None
                    st.session_state.next_question = None
                    st.session_state.top = None
                    st.rerun()
            with c2:
                if st.button("No"):
                    requests.post(f"{API}/answer", json={"session_id": sid, "symptom": st.session_state.next_symptom, "value": 0})
                    st.session_state.next_symptom = None
                    st.session_state.next_question = None
                    st.session_state.top = None
                    st.rerun()

# ---------- final output ----------
with st.container(border=True):
    st.subheader("4) Final output")
    if st.button("Get final output"):
        final = requests.get(f"{API}/final", params={"session_id": sid}).json()

        st.write("### Most likely matches (educational only)")
        for d, p in zip(final["top_diseases"], final["top_probs"]):
            st.write(f"- **{d}**: {p:.3f}")

        st.write("### Safe self-care")
        for tip in final["recommendations"]["self_care"]:
            st.write(f"- {tip}")

        st.write("### Warning signs — seek medical care")
        for red in final["recommendations"]["red_flags_seek_care"]:
            st.write(f"- {red}")

        st.warning(final["disclaimer"])