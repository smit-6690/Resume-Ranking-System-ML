import streamlit as st
import pandas as pd
import openai
import joblib
import os
import re
import tempfile
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

# Load model + BERT
model = joblib.load("models/best_model_rf.pkl")
bert_model = SentenceTransformer("all-MiniLM-L6-v2")
openai.api_key = "sk-..."  # Replace with your key or use environment var

# Text extract function
def extract_text(file):
    if file.name.endswith('.pdf'):
        reader = PdfReader(file)
        return " ".join([page.extract_text() or "" for page in reader.pages])
    return file.read().decode("utf-8")

# Clean text
def clean_text(text):
    text = re.sub(r"\n|\r|\t", " ", text.lower())
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return " ".join(text.split())

# Prompt template
def build_prompt(resume, jd):
    return f"""
You are an AI recruiter.

Here is the job description:
{jd}

Here is the candidate's resume:
{resume}

Based on this, generate 5 technical interview questions tailored to this candidate.
Return only the questions in numbered list format.
"""

# GPT generation
def generate_questions(resume, jd):
    prompt = build_prompt(resume, jd)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )
    return response['choices'][0]['message']['content']

# Feature builder
def extract_features(resume_clean, jd_clean):
    similarity = cosine_similarity(
        bert_model.encode([resume_clean]),
        bert_model.encode([jd_clean])
    )[0][0]

    resume_len = len(resume_clean.split())
    jd_words = set(jd_clean.split())
    resume_words = set(resume_clean.split())
    skill_match = len(jd_words & resume_words) / len(jd_words) if jd_words else 0
    years_exp = max([int(m) for m in re.findall(r'(\d+)\s+years?', resume_clean)] + [0])

    return similarity, skill_match, resume_len, years_exp

# UI Layout
st.title("📁 Bulk Resume Matcher & Question Generator")

jd_file = st.file_uploader("Upload Job Description (.pdf or .txt)", type=["pdf", "txt"])
resumes = st.file_uploader("Upload Multiple Resumes", type=["pdf", "txt"], accept_multiple_files=True)
top_n = st.number_input("Select Top N Candidates to Display", min_value=1, max_value=100, value=5)

if jd_file and resumes:
    jd_text = clean_text(extract_text(jd_file))

    results = []

    with st.spinner("Processing resumes..."):
        for i, resume_file in enumerate(resumes):
            resume_text = extract_text(resume_file)
            resume_clean = clean_text(resume_text)
            similarity, skill_match, resume_len, years_exp = extract_features(resume_clean, jd_text)

            features = pd.DataFrame([{
                "Similarity_Score": similarity,
                "skill_match_ratio": skill_match,
                "resume_length": resume_len,
                "years_of_experience": years_exp
            }])

            prediction = model.predict(features)[0]
            label = "Shortlisted" if prediction == 1 else "Rejected"

            results.append({
                "File Name": resume_file.name,
                "Similarity": round(similarity, 2),
                "Skill Match": round(skill_match, 2),
                "Resume Length": resume_len,
                "Years Exp": years_exp,
                "Prediction": label,
                "Resume Text": resume_text
            })

    df = pd.DataFrame(results)
    top_df = df[df["Prediction"] == "Shortlisted"].sort_values(by="Similarity", ascending=False).head(top_n)

    st.success(f"✅ Top {top_n} Shortlisted Candidates")
    st.dataframe(top_df[['File Name', 'Similarity', 'Skill Match', 'Prediction']])

    if st.button("💬 Generate GPT Questions for Top Candidates"):
        with st.spinner("Generating questions..."):
            gpt_results = []
            for _, row in top_df.iterrows():
                try:
                    q = generate_questions(row["Resume Text"], jd_text)
                except:
                    q = "⚠️ Error generating questions."
                gpt_results.append({
                    "File Name": row["File Name"],
                    "Questions": q
                })
            q_df = pd.DataFrame(gpt_results)
            st.subheader("Interview Questions")
            st.write(q_df)
