import streamlit as st
import pandas as pd
import openai
import joblib
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

# Load your model
model = joblib.load("models/best_model_rf.pkl")

# Load BERT model
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# OpenAI Key setup
openai.api_key = "OPENAI_API_KEY "

# Function to extract text from uploaded file
def extract_text(file):
    if file.name.endswith('.pdf'):
        reader = PdfReader(file)
        text = " ".join([page.extract_text() or "" for page in reader.pages])
    else:
        text = file.read().decode("utf-8")
    return text

# Clean text
def clean_text(text):
    text = re.sub(r'\n|\r|\t', ' ', text.lower())
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return ' '.join(text.split())

# GPT prompt
def build_prompt(resume, jd):
    return f"""
You are an AI recruiter.

Here is the job description:
{jd}

Here is the candidate's resume:
{resume}

Based on this, generate 5 tailored interview questions to evaluate this candidate's fit for the role.
Return only the questions in numbered list format.
"""

def generate_questions(resume, jd):
    prompt = build_prompt(resume, jd)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )
    return response['choices'][0]['message']['content']

# Streamlit UI
st.title("📄 Resume & JD Matcher + Interview Question Generator")

resume_file = st.file_uploader("Upload Resume (.pdf or .txt)", type=["pdf", "txt"])
jd_file = st.file_uploader("Upload Job Description (.pdf or .txt)", type=["pdf", "txt"])

# Initialize session state variables
for key in ['features', 'prediction', 'label', 'questions', 'resume_raw', 'jd_raw']:
    if key not in st.session_state:
        st.session_state[key] = None

if resume_file and jd_file:
    # Extract and clean
    resume_raw = extract_text(resume_file)
    jd_raw = extract_text(jd_file)
    resume_clean = clean_text(resume_raw)
    jd_clean = clean_text(jd_raw)

    # Store raw versions for GPT later
    st.session_state.resume_raw = resume_raw
    st.session_state.jd_raw = jd_raw

    # Show previews
    st.subheader("Resume Preview:")
    st.write(resume_raw[:500])

    st.subheader("JD Preview:")
    st.write(jd_raw[:500])

    # BERT similarity
    resume_vec = bert_model.encode([resume_clean])
    jd_vec = bert_model.encode([jd_clean])
    similarity = cosine_similarity(resume_vec, jd_vec)[0][0]

    st.metric("📊 Semantic Similarity Score", f"{similarity:.2f}")

    # Feature extraction and store in session state
    resume_len = len(resume_clean.split())
    jd_keywords = set(jd_clean.split())
    resume_words = set(resume_clean.split())
    skill_match = len(jd_keywords & resume_words) / len(jd_keywords) if jd_keywords else 0
    years_exp = max([int(m) for m in re.findall(r'(\d+)\s+years?', resume_clean)] + [0])

    features = pd.DataFrame([{
        "Similarity_Score": similarity,
        "skill_match_ratio": skill_match,
        "resume_length": resume_len,
        "years_of_experience": years_exp
    }])

    st.session_state.features = features

# Prediction Button
if st.session_state.features is not None:
    if st.button("Get Model Prediction"):
        prediction = model.predict(st.session_state.features)[0]
        label = "✅ Shortlisted" if prediction == 1 else "❌ Rejected"
        st.session_state.prediction = prediction
        st.session_state.label = label

# Display prediction if available
if st.session_state.label:
    st.subheader(f"Model Prediction: {st.session_state.label}")

# Interview Button
if st.session_state.label:
    if st.session_state.prediction == 1:
        if st.button("Generate Interview Questions"):
            with st.spinner("Generating questions..."):
                questions = generate_questions(st.session_state.resume_raw, st.session_state.jd_raw)
                st.session_state.questions = questions
    else:
        st.button("Generate Interview Questions", disabled=True)

# Display questions
if st.session_state.questions:
    st.subheader("💬 Interview Questions")
    st.write(st.session_state.questions)


