from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import os



print("generating questions")
# The API key lives in a file literally named "_env" (not the usual ".env").
# Resolve it relative to THIS file's location, not the current working
# directory, so it works no matter where you launch `streamlit run` from.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv()

api_key = os.getenv("api_key_groq")
if not api_key:
    raise RuntimeError(
        f"api_key_groq not found. Make sure a file named '_env' exists at "
        f"{BASE_DIR} and contains a line like: api_key_groq=your_key_here"
    )

client = Groq(api_key=api_key)

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def send_to_llm(text, temperature=0.3):
    """Send resume + JD text to the LLM and get back a list of interview questions."""
    prompt = f"""
{text}

You are a helpful assistant that extracts relevant information from the given text.
Your task is to generate as many interview questions as possible based on the
resume and job description provided above. The questions should be relevant to
the job description and the candidate's resume.

Number every question (e.g. "1. ...", "2. ..."). Return ONLY the numbered
questions - no preamble, no explanations, no closing remarks.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )

    return response.choices[0].message.content.strip()