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
import aiofiles
import asyncio

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
    return encoder.encode(text, convert_to_tensor=False)


# Load FAQ and build lookup map
faq_data = []
faq_map = {}
question_bank = []
try:
    with open("faq4.json", "r") as f:
        faq_data = json.load(f)["intents"]
        for item in faq_data:
            intent_title = item.get("title", "").lower().replace(" ", "_")
            faq_map[intent_title] = item["answer"]
            for tag in item.get("tags", []):
                faq_map[tag.lower().replace(" ", "_")] = item["answer"]
            for q in item.get("questions", []):
                question_bank.append((get_cached_embedding(clean_text(q)), q, item["answer"]))
except Exception as e:
    print("❌ Error loading FAQ:", e)


# Cache Rasa responses
@lru_cache(maxsize=1000)
def get_rasa_intent(prompt):
    try:
        response = requests.post(
            RASA_NLU_ENDPOINT,
            json={"text": clean_text(prompt)},
            timeout=3
        ).json()
        return response.get("intent", {}).get("name"), response.get("intent", {}).get("confidence", 0)
    except Exception as e:
        print("❌ Rasa error:", e)
        return None, 0


# Async log interaction
async def log_interaction_async(prompt, response, source):
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
        async with aiofiles.open(logs_path, "r") as file:
            content = await file.read()
            if content:
                data = json.loads(content)
    data.append(log)
    async with aiofiles.open(logs_path, "w") as file:
        await file.write(json.dumps(data, indent=4))


# Stream response
def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.002)  # Reduced from 0.005 for faster streaming


# Semantic fallback with cosine similarity
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
    intent, confidence = get_rasa_intent(prompt)

    if confidence > 0.7 and intent:
        course_data = {
            "foundational_digital_skills_overview": (
                "The Foundational Digital Skills Curriculum empowers beginners with essential digital skills.",
                "Available courses include:\n"
                "- F101: Operating Digital Devices\n"
                "- F102: Making Social Connection\n"
                "- F103: Doing Business\n"
                "- F104: Accessing Government Services\n"
                "- F105: Cyber Hygiene for Everyone\n"
                "- F106: eWaste Management"
            ),
            "basic_digital_skills_overview": (
                "The Basic Digital Skills curriculum equips learners with essential skills for today's digital society.",
                "Available courses include:\n"
                "- B201: Digital Devices in Digital Workspace\n"
                "- B202: Skills in Basic Productivity Tools\n"
                "- B203: Communication and Collaboration\n"
                "- B204: Access to Government Services\n"
                "- B205: Video Production and Editing\n"
                "- B206: Learning Online\n"
                "- B207: Basic Online Work Skills\n"
                "- B208: Digital Entrepreneurship\n"
                "- B209: Cyberhygiene\n"
                "- B210: E-Waste Management"
            ),
            "intermediate_digital_skills_overview": (
                "The Intermediate Digital Skills courses provide in-depth expertise and career-ready proficiency.",
                "Available courses include:\n"
                "- I303: Advanced Excel for Analysis\n"
                "- I304: eLearning Content Production\n"
                "- I305: Data Management and Analytics\n"
                "- I306: Digital Marketing, E-Commerce & Entrepreneurship\n"
                "- I307: Web Design and Development\n"
                "- I308: Introduction to Programming with Python\n"
                "- I310: Fundamentals of Networking\n"
                "- I313: eWaste Management & Circularity\n"
                "- I314: Cyber Hygiene for SMEs"
            ),
            "advanced_digital_skills_overview": (
                "Advanced Digital Skills courses are designed for career-level technical expertise.",
                "Available courses include:\n"
                "- A401: AI for SDGs\n"
                "- A402: Fintech Technologies\n"
                "- A403: Cybersecurity and Analytics\n"
                "- A404: Integrated Circuit Design and Fabrication (IoT)\n"
                "- A405: Data Analytics for Decision Making\n"
                "- A406: Advanced Project Management\n"
                "- A407: eWaste Management and Circular AI\n"
                "- A408: Digital Project Monitoring and Evaluation (MEL)\n"
                "- A409: Project Contract Management\n"
                "- A410: Large Programme Management"
            )
        }

        if intent == "payment_info":
            # Combine payment-related responses
            payment_answers = {}
            for faq_item in faq_data:
                faq_title = faq_item.get("title", "").lower().replace(" ", "_")
                if faq_title in ["payment_process", "paybill_and_account_details", "course_fees", "payment_info"]:
                    payment_answers[faq_title] = faq_item["answer"]
            if payment_answers.get("payment_info"):
                combined_response = payment_answers["payment_info"]
            else:
                combined_response = (
                    f"{payment_answers.get('course_fees', 'Course fees information not available.')}\n\n"
                    f"Payment Process: {payment_answers.get('payment_process', 'Payments are made through the eCitizen platform.')}\n\n"
                    f"Paybill and Account Details: {payment_answers.get('paybill_and_account_details', 'Use Paybill number 222222 on eCitizen.')}"
                )
                if not payment_answers:
                    combined_response = (
                        "Payments for Smart Academy courses are made through the eCitizen platform using Paybill number 222222. "
                        "Account numbers are generated automatically per course during registration. Course fees are as follows:\n"
                        "- Foundational: 500 KES\n"
                        "- Basic: 2,500 KES\n"
                        "- Intermediate: 5,000 KES\n"
                        "- Advanced: 12,000 KES\n"
                        "Payments can be made via bank transfer or mobile money through eCitizen. Receipts are available on your Smart Academy portal after payment. "
                        "For more details, visit [invalid url, do not cite]."
                    )
            asyncio.run(log_interaction_async(prompt, combined_response, f"FAQ (combined payment_info)"))
            return combined_response

        if intent in course_data:
            overview, unit_list = course_data[intent]
            answer = faq_map.get(intent, overview)
            combined_response = f"{answer}\n\n{unit_list}"
            asyncio.run(log_interaction_async(prompt, combined_response, f"FAQ (combined {intent})"))
            return combined_response

        if intent in faq_map:
            answer = faq_map[intent]
            asyncio.run(log_interaction_async(prompt, answer, f"FAQ ({intent})"))
            return answer

    answer = get_multi_match(prompt)
    if answer:
        asyncio.run(log_interaction_async(prompt, answer, "Multi Semantic Match"))
        return answer

    fallback = (
        "I'm not confident I can help with that. Talk to an agent for further enquiries.\n\n"
        "📞 Contact Smart Academy Support Team:\n"
        "- Phone: +254 712 345 678\n"
        "- Email: support@smartacademy.go.ke\n"
        "- Website: https://smartacademy.go.ke"
    )
    asyncio.run(log_interaction_async(prompt, fallback, "Escalation"))
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
