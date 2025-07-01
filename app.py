import streamlit as st
import requests
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from functools import lru_cache

# Load environment and Gemini
load_dotenv()
genai.configure(api_key=os.getenv("API_KEY"))
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# RASA_NLU_ENDPOINT = "http://localhost:5005/model/parse"
RASA_NLU_ENDPOINT = "https://smartguide-rasa-production.up.railway.app/model/parse"

# --- Preprocess and cache embedding ---
def clean_text(text):
    return text.strip().lower().replace("?", "").replace(".", "")

@lru_cache(maxsize=1000)
def get_cached_embedding(text):
    return encoder.encode(text, convert_to_tensor=True)

# Load FAQ and cache embeddings
faq_data = []
question_bank = []
try:
    with open("faq3.json", "r") as f:
        faq_data = json.load(f)["intents"]
        for item in faq_data:
            for q in item.get("questions", []):
                vec = get_cached_embedding(clean_text(q))
                question_bank.append((vec, q, item["answer"]))
except Exception as e:
    print("❌ Error loading FAQ:", e)

# Log interaction
def log_interaction(prompt, response, source):
    log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": prompt,
        "response": response,
        "source": source
    }
    os.makedirs("logs", exist_ok=True)
    logs_path = "logs/chat_logs.json"
    data = []
    if os.path.exists(logs_path):
        with open(logs_path, "r") as file:
            data = json.load(file)
    data.append(log)
    with open(logs_path, "w") as file:
        json.dump(data, file, indent=4)

# Stream response
def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.005)

# Semantic fallback with multiple answers
def get_multi_match(prompt):
    query_vec = get_cached_embedding(clean_text(prompt))
    matches = []
    for vec, q, answer in question_bank:
        score = util.cos_sim(query_vec, vec).item()
        if score > 0.65:
            matches.append((score, answer))

    matches = sorted(matches, reverse=True)
    seen = set()
    responses = []
    for _, ans in matches:
        if ans not in seen:
            responses.append(ans)
            seen.add(ans)
        if len(responses) == 3:
            break

    return "\n\n".join(f"✅ {r}" for r in responses) if responses else None

# Get response logic
def get_response(prompt):
    cleaned = clean_text(prompt)

    try:
        rasa_response = requests.post(
            RASA_NLU_ENDPOINT,
            json={"text": cleaned},
            timeout=5
        ).json()

        intent = rasa_response.get("intent", {}).get("name")
        confidence = rasa_response.get("intent", {}).get("confidence", 0)

        if confidence > 0.7:
            for item in faq_data:
                title_match = item.get("title", "").lower().replace(" ", "_") == intent
                tag_match = intent in [tag.lower().replace(" ", "_") for tag in item.get("tags", [])]
                if title_match or tag_match:
                    log_interaction(prompt, item["answer"], f"FAQ ({intent})")
                    return item["answer"]
    except Exception as e:
        print("❌ Rasa error:", e)

    answer = get_multi_match(prompt)
    if answer:
        log_interaction(prompt, answer, "Multi Semantic Match")
        return answer

    fallback = (
        "I'm not confident I can help with that. Talk to an agent for further enquiries.\n\n"
        "📞 Contact Smart Academy Support Team:\n"
        "- Name: "
        "- Phone: +254 712 345 678\n"
        "- Email: support@smartacademy.go.ke\n"
        "- Website: https://smartacademy.go.ke"
    )
    log_interaction(prompt, fallback, "Escalation")
    return fallback

# Streamlit UI
st.set_page_config(page_title="Smart Academy Chatbot", page_icon="📝")
st.title("SmartGuide")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello 👋, how may I help you today?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything about Smart Academy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = get_response(prompt)
        st.write_stream(stream_data(response))
        st.session_state.messages.append({"role": "assistant", "content": response})
