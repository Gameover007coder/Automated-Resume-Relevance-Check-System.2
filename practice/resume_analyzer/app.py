# app.py
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os

from parser import parse_resume
from analyzer import analyze_resume
from database import init_db, save_evaluation, get_all_evaluations

# Load environment variables from .env file
load_dotenv()

# Check for API key
if not os.getenv("GOOGLE_API_KEY"):
    st.error("Google API key not found. Please set it in your .env file.")
    st.stop()

# --- App Layout ---
st.set_page_config(page_title="Automated Resume Relevance Checker", layout="wide")
st.title("🤖 Automated Resume Relevance Check System")

# Initialize the database
init_db()

# --- Main Application ---
tab1, tab2 = st.tabs(["Single Evaluation", "Past Evaluations"])

with tab1:
    st.header("Evaluate a New Resume")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Job Description")
        jd_text = st.text_area("Paste the job description here", height=300)
    
    with col2:
        st.subheader("📄 Upload Resume")
        uploaded_resume = st.file_uploader("Upload a single resume (PDF or DOCX)", type=["pdf", "docx"])

    if st.button("Analyze Resume", type="primary"):
        if not jd_text or not uploaded_resume:
            st.warning("Please provide both a job description and a resume.")
        else:
            with st.spinner("Processing... This may take a moment. 🦾"):
                try:
                    resume_text = parse_resume(uploaded_resume)
                    analysis_result = analyze_resume(resume_text, jd_text)
                    
                    st.success("Analysis Complete!")
                    st.subheader("Evaluation Results")
                    
                    score = analysis_result["relevance_score"]
                    verdict = analysis_result["verdict"]
                                        
                    st.metric(label="**Relevance Score**", value=f"{score}%")
                    st.subheader(f"Fit Verdict: **{verdict}**")

                    st.markdown("---")

                    with st.expander("**Missing Must-Have Skills**"):
                        st.markdown(f"> {analysis_result['missing_skills']}")

                    with st.expander("**Personalized Feedback for Candidate**"):
                        st.markdown(analysis_result["feedback"])

                    save_evaluation(
                        jd=jd_text,
                        resume_filename=uploaded_resume.name,
                        score=analysis_result["relevance_score"],
                        verdict=analysis_result["verdict"],
                        missing_skills=analysis_result["missing_skills"],
                        feedback=analysis_result["feedback"]
                    )
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")

with tab2:
    st.header("Past Evaluation Dashboard")
    st.write("Here you can see all previously processed resumes.")
    
    all_records = get_all_evaluations()
    
    if not all_records:
        st.info("No evaluations have been saved yet.")
    else:
        df = pd.DataFrame(all_records, columns=["Job Description", "Resume", "Score", "Verdict", "Missing Skills", "Feedback", "Timestamp"])
        st.dataframe(df[['Timestamp', 'Resume', 'Score', 'Verdict']], use_container_width=True)
        
        selected_index = st.selectbox("Select an evaluation to see full details", df.index, format_func=lambda x: f"{df.at[x, 'Timestamp']} - {df.at[x, 'Resume']}")
        if selected_index is not None:
            st.markdown("---")
            selected_row = df.loc[selected_index]
            st.subheader(f"Details for {selected_row['Resume']}")
            st.text(f"Evaluated on: {selected_row['Timestamp']}")
            st.text(f"Score: {selected_row['Score']}% - Verdict: {selected_row['Verdict']}")
            
            with st.expander("Original Job Description"):
                st.text(selected_row['Job Description'])
            with st.expander("Missing Skills Identified"):
                 st.text(selected_row['Missing Skills'])
            with st.expander("Feedback Provided"):
                 st.text(selected_row['Feedback'])