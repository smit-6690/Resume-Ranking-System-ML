# ML Resume Ranking & Interview Question Generator

This project is an end-to-end machine learning application for automated resume ranking and tailored interview question generation. It matches candidate resumes to job descriptions, predicts shortlist suitability, and generates interview questions using GPT.

## Features

- **Resume & JD Matching:** Uses BERT embeddings and feature engineering to compute similarity between resumes and job descriptions.
- **ML Model Prediction:** Predicts whether a candidate should be shortlisted using a trained Random Forest model.
- **Interview Question Generation:** Generates custom interview questions for shortlisted candidates using OpenAI GPT.
- **Bulk Processing:** Supports batch processing of multiple resumes.
- **Interactive UI:** Built with Streamlit for easy file uploads and results visualization.

## Project Structure

```
ML-project-main/
├── streamlit_app.py                # Main Streamlit app for single resume/JD
├── streamlit_app_batch.py          # Streamlit app for bulk processing
├── models/
│   └── best_model_rf.pkl           # Trained Random Forest model
├── data/
│   ├── resumes_dataset.csv         # Raw resumes
│   ├── resumes_cleaned.csv         # Cleaned resumes
│   ├── job_descriptions.csv        # Job descriptions
│   ├── interview_questions_output.csv # Generated interview questions
│   └── ...                         # Other data files
├── notebooks/
│   ├── 01_inspect_resume_dataset.ipynb
│   ├── 02_preprocess_resumes.ipynb
│   ├── 03_bert_resume_matching.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_model_training.ipynb
│   └── 07_interview_question_generator.ipynb
└── outputs/
    └── ...                         # Model comparison and other outputs
```

## Setup

1. **Clone the repository**
2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
3. **Set up OpenAI API key**
   - Add your OpenAI API key in `streamlit_app.py` and `streamlit_app_batch.py` or set it as an environment variable.

4. **Run the Streamlit app**
   ```sh
   streamlit run ML-project-main/streamlit_app.py
   ```
   For bulk processing:
   ```sh
   streamlit run ML-project-main/streamlit_app_batch.py
   ```

## Usage

- **Single Resume/JD:** Upload a resume and job description to get a match score, prediction, and interview questions.
- **Bulk Processing:** Upload a JD and multiple resumes to get top-N ranked candidates and questions.

## Model Training

- See [notebooks/06_model_training.ipynb](ML-project-main/notebooks/06_model_training.ipynb) for model training and evaluation.
- The best model is saved as [models/best_model_rf.pkl](ML-project-main/models/best_model_rf.pkl).

## Data

- Raw and cleaned resumes: [data/resumes_dataset.csv](ML-project-main/data/resumes_dataset.csv), [data/resumes_cleaned.csv](ML-project-main/data/resumes_cleaned.csv)
- Job descriptions: [data/job_descriptions.csv](ML-project-main/data/job_descriptions.csv)
- Generated interview questions: [data/interview_questions_output.csv](ML-project-main/data/interview_questions_output.csv)

## Notebooks

- Data inspection, preprocessing, feature engineering, model training, and question generation are documented in the [notebooks](ML-project-main/notebooks/) directory.
