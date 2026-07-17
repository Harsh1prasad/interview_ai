import os
import re
import streamlit as st

from question_generation import question_generation
from evaluator import ans_evaluation
from stt_tts.stt import SpeechToText
from stt_tts.tts import TextToSpeech

st.set_page_config(page_title="AI Interview Prep", layout="centered")
st.title("AI Interview Prep")

# ---- cached engines (loaded once, reused across reruns) ------------------
@st.cache_resource
def get_stt_engine():
    return SpeechToText(model_size="base")


@st.cache_resource
def get_tts_engine():
    return TextToSpeech()


# ---- session state setup ------------------------------------------------
if "questions" not in st.session_state:
    st.session_state.questions = []
if "current_q" not in st.session_state:
    st.session_state.current_q = 0
if "evaluations" not in st.session_state:
    st.session_state.evaluations = []
if "pending_transcript" not in st.session_state:
    st.session_state.pending_transcript = ""

# ---- step 1: upload resume + JD, generate questions ----------------------
resume = st.file_uploader("Upload resume (PDF)", type=["pdf"])
jd = st.file_uploader("Upload job description (PDF)", type=["pdf"])

if resume and jd and st.button("Generate questions"):
    with st.spinner("Generating interview questions..."):
        raw_questions = question_generation(resume, jd)
        questions = [q.strip() for q in raw_questions if re.match(r"^\d+[\.\)]", q.strip())]

    if not questions:
        st.error("Couldn't parse any questions from the model's response. Please try again.")
    else:
        st.session_state.questions = questions
        st.session_state.current_q = 0
        st.session_state.evaluations = []
        st.rerun()

# ---- step 2: walk through questions, collect + evaluate answers ----------
if st.session_state.questions:
    st.subheader("Interview")
    idx = st.session_state.current_q
    total = len(st.session_state.questions)

    if idx < total:
        question_text = st.session_state.questions[idx]
        # strip the leading "1. " / "2) " numbering before speaking it aloud
        clean_question = re.sub(r"^\d+[\.\)]\s*", "", question_text)

        st.write(f"**Question {idx + 1} of {total}**")
        st.write(question_text)

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button("🔊 Read question aloud", key=f"tts_{idx}"):
                tts = get_tts_engine()
                with st.spinner("Speaking..."):
                    tts.speak(clean_question)

        with col2:
            record_seconds = st.number_input(
                "Recording length (sec)",
                min_value=5,
                max_value=60,
                value=15,
                step=5,
                key=f"dur_{idx}",
            )

        with col3:
            if st.button("🎙️ Record answer", key=f"record_{idx}"):
                stt = get_stt_engine()
                audio_path = f"answer_{idx}.wav"
                with st.spinner(f"Recording for {record_seconds}s... speak now"):
                    stt.record_audio(duration=int(record_seconds), filename=audio_path)
                with st.spinner("Transcribing..."):
                    transcript = stt.transcribe(audio_path)
                # clean up the temp audio file
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                st.session_state.pending_transcript = transcript
                st.rerun()

        answer_key = f"answer_{idx}"
        # if a recording was just transcribed, seed the text area with it
        # (must be set BEFORE the widget is instantiated)
        if st.session_state.pending_transcript:
            st.session_state[answer_key] = st.session_state.pending_transcript
            st.session_state.pending_transcript = ""

        answer = st.text_area("Your answer", key=answer_key)

        if st.button("Submit answer", key=f"submit_{idx}"):
            if answer.strip():
                with st.spinner("Evaluating your answer..."):
                    feedback = ans_evaluation(question_text, answer)
                st.session_state.evaluations.append(
                    {
                        "question": question_text,
                        "answer": answer,
                        "feedback": feedback,
                    }
                )
                st.session_state.current_q += 1
                st.rerun()
            else:
                st.warning("Please enter an answer before submitting.")
    else:
        st.success("Interview complete! See your feedback below.")

# ---- step 3: feedback summary --------------------------------------------
if st.session_state.evaluations:
    st.subheader("Feedback summary")
    for i, e in enumerate(st.session_state.evaluations, start=1):
        with st.expander(f"Q{i}: {e['question']}"):
            st.write(f"**Your answer:** {e['answer']}")
            st.write("**Feedback:**")
            st.write(e["feedback"])