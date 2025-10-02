import streamlit as st
import re
import numpy as np
import pandas as pd
from datetime import datetime
import sqlite3
import tempfile
import os

# Import with error handling
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    st.warning("pdfplumber not installed. PDF parsing will be disabled.")

try:
    import docx2txt
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    st.warning("docx2txt not installed. DOCX parsing will be disabled.")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    NLTK_SUPPORT = True
except ImportError:
    NLTK_SUPPORT = False
    st.warning("nltk not installed. Text processing will be limited.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_SUPPORT = True
except ImportError:
    SKLEARN_SUPPORT = False
    st.warning("scikit-learn not installed. Semantic matching will be disabled.")

try:
    import plotly.express as px
    PLOTLY_SUPPORT = True
except ImportError:
    PLOTLY_SUPPORT = False
    st.warning("plotly not installed. Charts will be disabled.")

class ResumeAnalyzer:
    def __init__(self):
        self.stop_words = set()
        self.skill_keywords = self._load_skill_keywords()
        self._setup_nltk()
        
    def _setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set(['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        
    def _load_skill_keywords(self):
        """Load common technical skills"""
        common_skills = [
            'python', 'java', 'javascript', 'sql', 'html', 'css', 'machine learning', 
            'deep learning', 'data analysis', 'tableau', 'power bi', 'excel', 
            'project management', 'agile', 'scrum', 'devops', 'aws', 'azure', 
            'google cloud', 'docker', 'kubernetes', 'react', 'angular', 'vue', 
            'node.js', 'django', 'flask', 'fastapi', 'tensorflow', 'pytorch', 
            'natural language processing', 'computer vision', 'ci/cd', 'git', 
            'jenkins', 'rest api', 'graphql', 'mongodb', 'postgresql', 'mysql',
            'c++', 'c#', 'php', 'ruby', 'swift', 'kotlin', 'typescript',
            'data science', 'big data', 'hadoop', 'spark', 'kafka', 'redis',
            'linux', 'unix', 'windows', 'macos', 'android', 'ios'
        ]
        return set(common_skills)
    
    def extract_text_from_file(self, file):
        """Extract text from PDF or DOCX files"""
        text = ""
        
        if file.type == "application/pdf":
            if not PDF_SUPPORT:
                st.error("PDF parsing is not available. Please install pdfplumber.")
                return ""
            try:
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e:
                st.error(f"Error reading PDF: {str(e)}")
                return ""
                
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            if not DOCX_SUPPORT:
                st.error("DOCX parsing is not available. Please install docx2txt.")
                return ""
            try:
                text = docx2txt.process(file)
            except Exception as e:
                st.error(f"Error reading DOCX: {str(e)}")
                return ""
        else:
            st.error("Unsupported file format. Please upload PDF or DOCX.")
            return ""
            
        return text if text.strip() else "No text could be extracted from the file."
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits, keep letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not NLTK_SUPPORT:
            # Simple tokenization without NLTK
            tokens = text.split()
            tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
            return " ".join(tokens)
        
        # Use NLTK if available
        try:
            tokens = word_tokenize(text)
            tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
            return " ".join(tokens)
        except:
            # Fallback to simple tokenization
            tokens = text.split()
            tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
            return " ".join(tokens)
    
    def extract_skills(self, text):
        """Extract skills from text"""
        found_skills = []
        text_lower = text.lower()
        for skill in self.skill_keywords:
            if skill in text_lower:
                found_skills.append(skill)
        return found_skills
    
    def extract_education(self, text):
        """Extract education information"""
        education_keywords = ['bachelor', 'master', 'phd', 'degree', 'diploma', 'certificate', 
                             'bs', 'ms', 'mba', 'btech', 'mtech', 'bsc', 'msc', 'be', 'me']
        sentences = re.split(r'[.!?]', text)
        education_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in education_keywords):
                education_sentences.append(sentence.strip())
        return education_sentences
    
    def calculate_hard_match_score(self, resume_text, jd_text):
        """Calculate hard match score based on keyword matching"""
        if not resume_text or not jd_text:
            return 0
            
        resume_tokens = set(self.preprocess_text(resume_text).split())
        jd_tokens = set(self.preprocess_text(jd_text).split())
        
        if not jd_tokens:
            return 0
            
        intersection = resume_tokens.intersection(jd_tokens)
        return min(len(intersection) / len(jd_tokens) * 100, 100)
    
    def calculate_semantic_similarity(self, resume_text, jd_text):
        """Calculate semantic similarity using TF-IDF and cosine similarity"""
        if not SKLEARN_SUPPORT or not resume_text or not jd_text:
            # Fallback to simple ratio if sklearn not available
            resume_words = set(resume_text.lower().split())
            jd_words = set(jd_text.lower().split())
            if not jd_words:
                return 0
            common_words = resume_words.intersection(jd_words)
            return len(common_words) / len(jd_words) * 100
        
        try:
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return min(cosine_sim[0][0] * 100, 100)
        except:
            # Fallback to simple ratio
            resume_words = set(resume_text.lower().split())
            jd_words = set(jd_text.lower().split())
            if not jd_words:
                return 0
            common_words = resume_words.intersection(jd_words)
            return len(common_words) / len(jd_words) * 100
    
    def calculate_final_score(self, hard_score, semantic_score, hard_weight=0.6, semantic_weight=0.4):
        """Calculate weighted final score"""
        return (hard_score * hard_weight) + (semantic_score * semantic_weight)
    
    def get_missing_elements(self, resume_text, jd_text):
        """Identify missing skills/qualifications from JD"""
        jd_skills = self.extract_skills(jd_text)
        resume_skills = self.extract_skills(resume_text)
        missing_skills = [skill for skill in jd_skills if skill not in resume_skills]
        
        # Extract education requirements from JD
        jd_education = self.extract_education(jd_text)
        resume_education = self.extract_education(resume_text)
        missing_education = []
        
        # Simple check for education gaps
        if jd_education and not resume_education:
            missing_education = ["Required education qualifications not found in resume"]
        elif jd_education:
            # Check if resume has similar education terms
            jd_edu_terms = set()
            for edu in jd_education:
                jd_edu_terms.update(edu.lower().split())
            
            resume_edu_terms = set()
            for edu in resume_education:
                resume_edu_terms.update(edu.lower().split())
                
            missing_edu_terms = jd_edu_terms - resume_edu_terms
            if missing_edu_terms:
                missing_education = list(missing_edu_terms)
            
        return missing_skills, missing_education
    
    def get_verdict(self, score):
        """Get suitability verdict based on score"""
        if score >= 75:
            return "High suitability"
        elif score >= 50:
            return "Medium suitability"
        else:
            return "Low suitability"
    
    def generate_feedback(self, missing_skills, missing_education, score):
        """Generate personalized feedback for improvement"""
        feedback = []
        
        if score < 50:
            feedback.append("Your resume needs significant improvements to match job requirements.")
        elif score < 75:
            feedback.append("Your resume has some alignment but could be improved.")
        else:
            feedback.append("Your resume is well-aligned with the job requirements.")
        
        if missing_skills:
            feedback.append(f"Consider adding these skills: {', '.join(missing_skills[:5])}")  # Show top 5
        
        if missing_education:
            feedback.append(f"Consider highlighting these qualifications: {', '.join(missing_education[:3])}")
        
        if score < 75:
            feedback.append("Consider adding more relevant projects and experiences.")
            feedback.append("Use more keywords from the job description in your resume.")
        
        if not feedback:
            feedback.append("Your resume looks good! Keep up the good work.")
            
        return feedback

class DatabaseManager:
    def __init__(self, db_name="resume_evaluations.db"):
        self.db_name = db_name
        self.init_db()
    
    def init_db(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        
        # Create evaluations table
        c.execute('''CREATE TABLE IF NOT EXISTS evaluations
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      student_name TEXT,
                      job_title TEXT,
                      hard_score REAL,
                      semantic_score REAL,
                      final_score REAL,
                      verdict TEXT,
                      missing_skills TEXT,
                      missing_education TEXT,
                      feedback TEXT,
                      evaluation_date TIMESTAMP)''')
        
        # Create jobs table
        c.execute('''CREATE TABLE IF NOT EXISTS jobs
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      job_title TEXT,
                      job_description TEXT,
                      upload_date TIMESTAMP)''')
        
        conn.commit()
        conn.close()
    
    def save_evaluation(self, evaluation_data):
        """Save evaluation results to database"""
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        
        c.execute('''INSERT INTO evaluations 
                     (student_name, job_title, hard_score, semantic_score, final_score, 
                      verdict, missing_skills, missing_education, feedback, evaluation_date)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (evaluation_data['student_name'],
                  evaluation_data['job_title'],
                  evaluation_data['hard_score'],
                  evaluation_data['semantic_score'],
                  evaluation_data['final_score'],
                  evaluation_data['verdict'],
                  ', '.join(evaluation_data['missing_skills']),
                  ', '.join(evaluation_data['missing_education']),
                  ' | '.join(evaluation_data['feedback']),
                  datetime.now()))
        
        conn.commit()
        conn.close()
    
    def save_job(self, job_title, job_description):
        """Save job description to database"""
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        
        c.execute('''INSERT INTO jobs (job_title, job_description, upload_date)
                     VALUES (?, ?, ?)''',
                 (job_title, job_description, datetime.now()))
        
        conn.commit()
        conn.close()
    
    def get_evaluations(self):
        """Retrieve all evaluations from database"""
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        
        c.execute('''SELECT * FROM evaluations ORDER BY evaluation_date DESC''')
        evaluations = c.fetchall()
        
        conn.close()
        return evaluations
    
    def get_jobs(self):
        """Retrieve all jobs from database"""
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        
        c.execute('''SELECT * FROM jobs ORDER BY upload_date DESC''')
        jobs = c.fetchall()
        
        conn.close()
        return jobs

def main():
    st.set_page_config(
        page_title="Resume Relevance Check System",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 Automated Resume Relevance Check System")
    st.markdown("---")
    
    # Initialize analyzer and database
    analyzer = ResumeAnalyzer()
    db_manager = DatabaseManager()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    menu = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "📋 Upload Job", "📄 Evaluate Resume", "📊 View Results", "📈 Dashboard"]
    )
    
    if menu == "🏠 Home":
        show_home_page()
        
    elif menu == "📋 Upload Job":
        show_upload_job_page(db_manager)
        
    elif menu == "📄 Evaluate Resume":
        show_evaluate_resume_page(analyzer, db_manager)
        
    elif menu == "📊 View Results":
        show_view_results_page(db_manager)
        
    elif menu == "📈 Dashboard":
        show_dashboard_page(db_manager)

def show_home_page():
    st.header("Welcome to the Resume Relevance Check System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🚀 System Overview
        
        This AI-powered system automates resume evaluation against job requirements:
        
        - **Automated Scoring**: Get relevance scores (0-100) for each resume
        - **Gap Analysis**: Identify missing skills, certifications, and qualifications
        - **Smart Feedback**: Receive personalized improvement suggestions
        - **Consistent Evaluation**: Eliminate manual bias and inconsistencies
        - **Scalable Processing**: Handle thousands of resumes efficiently
        
        ### 📋 How to Use
        
        1. **Upload Job Descriptions**: Go to 'Upload Job' to add new job requirements
        2. **Evaluate Resumes**: Use 'Evaluate Resume' to analyze resumes against jobs
        3. **View Results**: Check all evaluations in 'View Results'
        4. **Analytics**: See trends and insights in 'Dashboard'
        """)
    
    with col2:
        st.info("**Quick Stats**")
        # You can add actual stats here later
        st.metric("Supported File Types", "PDF, DOCX")
        st.metric("Evaluation Metrics", "Hard + Semantic Match")
        st.metric("Output", "Score + Feedback + Verdict")
        
        st.warning("**Requirements**")
        st.write("""
        - Python 3.8+
        - Required packages installed
        - PDF/DOCX files for analysis
        """)

def show_upload_job_page(db_manager):
    st.header("📋 Upload Job Description")
    
    with st.form("job_upload_form"):
        job_title = st.text_input("Job Title*", placeholder="e.g., Senior Data Scientist")
        job_description = st.text_area(
            "Job Description*", 
            height=300,
            placeholder="Paste the complete job description here...\n\nInclude:\n- Required skills\n- Qualifications\n- Responsibilities\n- Experience requirements"
        )
        
        submitted = st.form_submit_button("💾 Save Job Description")
        
        if submitted:
            if not job_title.strip():
                st.error("Please enter a job title")
            elif not job_description.strip():
                st.error("Please enter a job description")
            else:
                db_manager.save_job(job_title.strip(), job_description.strip())
                st.success("✅ Job description saved successfully!")
                
                # Show preview
                with st.expander("Preview Saved Job"):
                    st.subheader(job_title)
                    st.write(job_description[:500] + "..." if len(job_description) > 500 else job_description)

def show_evaluate_resume_page(analyzer, db_manager):
    st.header("📄 Evaluate Resume Against Job")
    
    # Get available jobs
    jobs = db_manager.get_jobs()
    
    if not jobs:
        st.warning("🚨 No job descriptions found. Please upload a job description first.")
        return
    
    job_options = {f"{job[1]} (ID: {job[0]})": job[2] for job in jobs}
    selected_job_label = st.selectbox("Select Job Description*", list(job_options.keys()))
    jd_text = job_options[selected_job_label]
    
    st.subheader("Resume Information")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        student_name = st.text_input("Student Name*", placeholder="Enter student's full name")
    
    with col2:
        resume_file = st.file_uploader(
            "Upload Resume*", 
            type=["pdf", "docx"],
            help="Supported formats: PDF, DOCX"
        )
    
    if st.button("🔍 Evaluate Resume", type="primary"):
        if not student_name.strip():
            st.error("Please enter student name")
        elif not resume_file:
            st.error("Please upload a resume file")
        else:
            with st.spinner("Analyzing resume... This may take a few seconds."):
                try:
                    # Extract text from resume
                    resume_text = analyzer.extract_text_from_file(resume_file)
                    
                    if not resume_text or "No text could be extracted" in resume_text:
                        st.error("Could not extract text from the resume. Please try another file.")
                        return
                    
                    # Calculate scores
                    hard_score = analyzer.calculate_hard_match_score(resume_text, jd_text)
                    semantic_score = analyzer.calculate_semantic_similarity(resume_text, jd_text)
                    final_score = analyzer.calculate_final_score(hard_score, semantic_score)
                    
                    # Get missing elements
                    missing_skills, missing_education = analyzer.get_missing_elements(resume_text, jd_text)
                    
                    # Get verdict and feedback
                    verdict = analyzer.get_verdict(final_score)
                    feedback = analyzer.generate_feedback(missing_skills, missing_education, final_score)
                    
                    # Save evaluation
                    evaluation_data = {
                        'student_name': student_name,
                        'job_title': selected_job_label.split(" (ID:")[0],
                        'hard_score': round(hard_score, 2),
                        'semantic_score': round(semantic_score, 2),
                        'final_score': round(final_score, 2),
                        'verdict': verdict,
                        'missing_skills': missing_skills,
                        'missing_education': missing_education,
                        'feedback': feedback
                    }
                    db_manager.save_evaluation(evaluation_data)
                    
                    # Display results
                    display_evaluation_results(evaluation_data, resume_text, jd_text)
                    
                except Exception as e:
                    st.error(f"An error occurred during evaluation: {str(e)}")

def display_evaluation_results(evaluation_data, resume_text, jd_text):
    st.success("✅ Evaluation Complete!")
    
    # Score cards
    st.subheader("📊 Evaluation Scores")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Hard Match Score", 
            f"{evaluation_data['hard_score']:.1f}%",
            help="Based on keyword matching and exact skills"
        )
    
    with col2:
        st.metric(
            "Semantic Score", 
            f"{evaluation_data['semantic_score']:.1f}%",
            help="Based on contextual understanding and meaning"
        )
    
    with col3:
        st.metric(
            "Final Score", 
            f"{evaluation_data['final_score']:.1f}%",
            evaluation_data['verdict'],
            delta_color="off" if evaluation_data['final_score'] < 50 else "normal"
        )
    
    # Verdict with color coding
    st.subheader("🎯 Suitability Verdict")
    verdict = evaluation_data['verdict']
    if verdict == "High suitability":
        st.success(f"**{verdict}** - Strong match with job requirements")
    elif verdict == "Medium suitability":
        st.warning(f"**{verdict}** - Moderate match, some improvements needed")
    else:
        st.error(f"**{verdict}** - Significant improvements required")
    
    # Missing elements
    col1, col2 = st.columns(2)
    
    with col1:
        if evaluation_data['missing_skills']:
            st.subheader("🔧 Missing Skills")
            for skill in evaluation_data['missing_skills'][:10]:  # Show top 10
                st.write(f"- {skill}")
        else:
            st.subheader("✅ Skills Match")
            st.success("All required skills found in resume!")
    
    with col2:
        if evaluation_data['missing_education']:
            st.subheader("🎓 Missing Qualifications")
            for edu in evaluation_data['missing_education'][:5]:  # Show top 5
                st.write(f"- {edu}")
        else:
            st.subheader("✅ Qualifications Match")
            st.success("Education requirements met!")
    
    # Improvement feedback
    st.subheader("💡 Improvement Suggestions")
    for i, item in enumerate(evaluation_data['feedback'], 1):
        st.write(f"{i}. {item}")
    
    # Preview sections
    with st.expander("📝 Resume Text Preview"):
        st.text_area("Extracted Resume Text", resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text, height=200)
    
    with st.expander("📋 Job Description Preview"):
        st.text_area("Job Description", jd_text[:1000] + "..." if len(jd_text) > 1000 else jd_text, height=200)

def show_view_results_page(db_manager):
    st.header("📊 Evaluation Results")
    
    evaluations = db_manager.get_evaluations()
    
    if not evaluations:
        st.info("No evaluations found yet. Start by evaluating some resumes!")
        return
    
    # Convert to DataFrame
    eval_df = pd.DataFrame(evaluations, columns=[
        'ID', 'Student Name', 'Job Title', 'Hard Score', 'Semantic Score', 
        'Final Score', 'Verdict', 'Missing Skills', 'Missing Education', 
        'Feedback', 'Evaluation Date'
    ])
    
    # Filters
    st.subheader("🔍 Filter Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        job_filter = st.multiselect(
            "Filter by Job Title",
            options=eval_df['Job Title'].unique(),
            default=[]
        )
    
    with col2:
        verdict_filter = st.multiselect(
            "Filter by Verdict",
            options=eval_df['Verdict'].unique(),
            default=[]
        )
    
    with col3:
        score_range = st.slider(
            "Filter by Final Score",
            min_value=0,
            max_value=100,
            value=(0, 100)
        )
    
    # Apply filters
    filtered_df = eval_df.copy()
    
    if job_filter:
        filtered_df = filtered_df[filtered_df['Job Title'].isin(job_filter)]
    
    if verdict_filter:
        filtered_df = filtered_df[filtered_df['Verdict'].isin(verdict_filter)]
    
    filtered_df = filtered_df[
        (filtered_df['Final Score'] >= score_range[0]) & 
        (filtered_df['Final Score'] <= score_range[1])
    ]
    
    # Display results
    st.subheader(f"📋 Results ({len(filtered_df)} evaluations)")
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total", len(filtered_df))
    with col2:
        st.metric("High Suitability", len(filtered_df[filtered_df['Verdict'] == 'High suitability']))
    with col3:
        st.metric("Average Score", f"{filtered_df['Final Score'].mean():.1f}%")
    with col4:
        st.metric("Latest", filtered_df['Evaluation Date'].iloc[0].split()[0] if len(filtered_df) > 0 else "N/A")
    
    # Data table
    st.dataframe(
        filtered_df[[
            'Student Name', 'Job Title', 'Final Score', 'Verdict', 'Evaluation Date'
        ]].sort_values('Final Score', ascending=False),
        use_container_width=True
    )
    
    # Download option
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name=f"resume_evaluations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )
    
    # Detailed view
    with st.expander("🔍 View Detailed Results"):
        for _, eval_row in filtered_df.iterrows():
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{eval_row['Student Name']}** - {eval_row['Job Title']}")
                    st.write(f"Final Score: {eval_row['Final Score']}% | {eval_row['Verdict']}")
                with col2:
                    st.write(f"Date: {eval_row['Evaluation Date']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if eval_row['Missing Skills']:
                        st.write("Missing Skills:", eval_row['Missing Skills'])
                with col2:
                    if eval_row['Missing Education']:
                        st.write("Missing Education:", eval_row['Missing Education'])
                
                st.markdown("---")

def show_dashboard_page(db_manager):
    st.header("📈 Analytics Dashboard")
    
    evaluations = db_manager.get_evaluations()
    
    if not evaluations:
        st.info("No evaluation data available for analytics. Start by evaluating some resumes!")
        return
    
    eval_df = pd.DataFrame(evaluations, columns=[
        'ID', 'Student Name', 'Job Title', 'Hard Score', 'Semantic Score', 
        'Final Score', 'Verdict', 'Missing Skills', 'Missing Education', 
        'Feedback', 'Evaluation Date'
    ])
    
    # Convert date column
    eval_df['Evaluation Date'] = pd.to_datetime(eval_df['Evaluation Date'])
    eval_df['Date'] = eval_df['Evaluation Date'].dt.date
    
    # Overall statistics
    st.subheader("📊 Overall Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Evaluations", len(eval_df))
    
    with col2:
        avg_score = eval_df['Final Score'].mean()
        st.metric("Average Final Score", f"{avg_score:.1f}%")
    
    with col3:
        high_suitability = len(eval_df[eval_df['Verdict'] == 'High suitability'])
        st.metric("High Suitability", high_suitability)
    
    with col4:
        unique_students = eval_df['Student Name'].nunique()
        st.metric("Unique Students", unique_students)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Score Distribution")
        fig = px.histogram(
            eval_df, 
            x='Final Score',
            nbins=20,
            title="Distribution of Final Scores"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Verdict Distribution")
        verdict_counts = eval_df['Verdict'].value_counts()
        fig = px.pie(
            values=verdict_counts.values,
            names=verdict_counts.index,
            title="Suitability Verdict Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Trends over time
    st.subheader("📈 Trends Over Time")
    
    if len(eval_df) > 1:
        daily_avg = eval_df.groupby('Date')['Final Score'].mean().reset_index()
        daily_count = eval_df.groupby('Date').size().reset_index(name='Count')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Average Score Over Time")
            st.line_chart(daily_avg.set_index('Date')['Final Score'])
        
        with col2:
            st.write("Evaluations Per Day")
            st.bar_chart(daily_count.set_index('Date')['Count'])
    else:
        st.info("Need more data to show trends over time")
    
    # Job-specific analytics
    st.subheader("🔧 Job-specific Analysis")
    
    job_stats = eval_df.groupby('Job Title').agg({
        'Final Score': ['count', 'mean', 'std'],
        'Verdict': lambda x: (x == 'High suitability').sum()
    }).round(2)
    
    job_stats.columns = ['Count', 'Avg Score', 'Std Dev', 'High Suitability']
    job_stats['Success Rate'] = (job_stats['High Suitability'] / job_stats['Count'] * 100).round(1)
    
    st.dataframe(job_stats, use_container_width=True)

if __name__ == "__main__":
    main()