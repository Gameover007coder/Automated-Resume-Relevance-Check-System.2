# database.py
import sqlite3
import os

DB_FILE = "resume_evaluations.db"

def init_db():
    """Initializes the database and creates the table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_description TEXT NOT NULL,
            resume_filename TEXT NOT NULL,
            relevance_score INTEGER NOT NULL,
            fit_verdict TEXT NOT NULL,
            missing_skills TEXT,
            feedback TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized.")

def save_evaluation(jd, resume_filename, score, verdict, missing_skills, feedback):
    """Saves a single evaluation record to the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO evaluations (job_description, resume_filename, relevance_score, fit_verdict, missing_skills, feedback)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (jd, resume_filename, score, verdict, missing_skills, feedback))
    
    conn.commit()
    conn.close()

def get_all_evaluations():
    """Retrieves all evaluation records from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT job_description, resume_filename, relevance_score, fit_verdict, missing_skills, feedback, timestamp FROM evaluations ORDER BY timestamp DESC")
    records = cursor.fetchall()
    
    conn.close()
    return records