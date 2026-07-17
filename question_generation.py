
from parser.resume_parser import resume_text_extraction as resume
from parser.jd_parser import jd_text_extraction as jd
from llm_service.llm_service import send_to_llm
 
print("sending extracted text from jd and resume to llm")
def question_generation(upload_resume, upload_jd):
    resume_text = resume(upload_resume)
    jd_text = jd(upload_jd)
    response = send_to_llm(resume_text + "\n" + jd_text)
    questions = response.split("\n")
    return questions
 
