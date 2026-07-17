from llm_service.llm_service import client, MODEL

print("evaluating answer in evaluator")

def ans_evaluation(question, ans, temperature=0.3):
    """Evaluate a candidate's answer to an interview question and return feedback."""
    prompt = f"""You are an interviewer.

Question: {question}

Candidate's answer: {ans}

Rate this answer on the basis of:
1. Technical accuracy
2. Completeness
3. Communication

Give constructive feedback, then finish with scores in exactly this format:
Technical Accuracy: X/10
Completeness: X/10
Communication: X/10
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )

    return response.choices[0].message.content.strip()


