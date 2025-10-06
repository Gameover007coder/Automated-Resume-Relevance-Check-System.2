# analyzer.py (no spaCy version)
import os
import re
import json
import requests
from typing import List

# Ensure the API key is available
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Google API key not found. Please set it in your .env file.")

# Function to call Gemini API using requests
def call_gemini_api(prompt_text):
    api_key = os.getenv("GOOGLE_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "contents": [{
            "parts": [{
                "text": prompt_text
            }]
        }]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        raise Exception(f"Error calling Gemini API: {str(e)}")

def parse_jd_skills(jd_text: str):
    try:
        prompt = f"""
        You are an expert HR assistant. Extract key skills from the job description.
        Format your output as a JSON object with 'must_have_skills' and 'good_to_have_skills' keys.
        
        Job Description: {jd_text}
        
        Return ONLY valid JSON, no other text.
        """
        
        response = call_gemini_api(prompt)
        json_str = extract_json_from_text(response)
        data = json.loads(json_str)
        
        return {
            "must_have_skills": data.get('must_have_skills', []),
            "good_to_have_skills": data.get('good_to_have_skills', [])
        }
    except Exception as e:
        raise Exception(f"Error parsing job description skills: {str(e)}")

def perform_hard_match(resume_text: str, skills: dict) -> dict:
    try:
        # Simple text-based matching instead of spaCy tokenization
        resume_lower = resume_text.lower()
        
        must_have_found = []
        for skill in skills["must_have_skills"]:
            # Check if skill is in resume (case insensitive)
            if skill.lower() in resume_lower:
                must_have_found.append(skill)
        
        good_to_have_found = []
        for skill in skills["good_to_have_skills"]:
            # Check if skill is in resume (case insensitive)
            if skill.lower() in resume_lower:
                good_to_have_found.append(skill)
        
        must_have_missing = list(set(skills["must_have_skills"]) - set(must_have_found))
        
        score = 0
        if skills["must_have_skills"]:
            score += (len(must_have_found) / len(skills["must_have_skills"])) * 60
        if skills["good_to_have_skills"]:
            score += (len(good_to_have_found) / len(skills["good_to_have_skills"])) * 40
            
        return {"score": int(score), "must_have_missing": must_have_missing}
    except Exception as e:
        raise Exception(f"Error performing hard match: {str(e)}")

def extract_json_from_text(text):
    """Extract JSON from text that might contain markdown code blocks"""
    # Try to find JSON in markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    # Try to find JSON without markdown
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    return text

def generate_feedback_and_verdict(resume_text: str, jd_text: str, hard_match_results: dict) -> dict:
    try:
        prompt = f"""
        As an expert career coach, evaluate the following resume against the job description.
        Job Description: --- {jd_text} ---
        Resume Summary: --- {resume_text[:4000]} ---
        The initial keyword analysis identified these missing must-have skills: {', '.join(hard_match_results['must_have_missing']) or "None"}.
        Based on everything, provide:
        1. A final verdict: "High Fit", "Medium Fit", or "Low Fit".
        2. Constructive, personalized feedback for the candidate.
        Format your response as a JSON object with "verdict" and "feedback" keys.
        Return ONLY valid JSON, no other text.
        """
        
        response = call_gemini_api(prompt)
        json_str = extract_json_from_text(response)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # If JSON parsing fails, return a default response
            return {
                "verdict": "Medium Fit", 
                "feedback": "Could not parse detailed feedback. Please review manually."
            }
    except Exception as e:
        raise Exception(f"Error generating feedback: {str(e)}")

def analyze_resume(resume_text: str, jd_text: str) -> dict:
    try:
        parsed_skills = parse_jd_skills(jd_text)
        hard_match_results = perform_hard_match(resume_text, parsed_skills)
        
        # For semantic matching, we'll use a simpler approach
        semantic_prompt = f"""
        Rate the similarity between this resume and job description on a scale of 0-100.
        Resume: {resume_text[:3000]}
        Job Description: {jd_text[:2000]}
        Return only a number between 0-100, no other text.
        """
        
        semantic_response = call_gemini_api(semantic_prompt)
        try:
            semantic_score = int(re.search(r'\d+', semantic_response).group(0))
        except:
            semantic_score = 50  # Default if parsing fails
        
        final_score = int((hard_match_results['score'] * 0.4) + (semantic_score * 0.6))
        feedback_and_verdict = generate_feedback_and_verdict(resume_text, jd_text, hard_match_results)

        return {
            "relevance_score": final_score,
            "verdict": feedback_and_verdict.get("verdict", "N/A"),
            "missing_skills": ", ".join(hard_match_results['must_have_missing']) or "None",
            "feedback": feedback_and_verdict.get("feedback", "N/A")
        }
    except Exception as e:
        raise Exception(f"Error analyzing resume: {str(e)}")