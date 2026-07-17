import os
import re
import streamlit as st

from question_generation import question_generation
from evaluator import ans_evaluation
from stt_tts.stt import SpeechToText, MIC_AVAILABLE
from stt_tts.tts import TextToSpeech

st.set_page_config(page_title="AI Interview Prep", layout="centered")
st.title("AI Interview Prep")

# ---- cached engines (loaded once, reused across reruns) ------------------
# NOTE: SpeechToText() loading a Whisper model and TextToSpeech() trying to
# init a system engine can both still fail for reasons other than "no mic"
# (e.g. out of memory, no espeak binary). Wrap in try/except too, so a
# surprise failure degrades the feature instead of crashing the app.
@st.cache_resource
def get_stt_engine():
    try:
        return SpeechToText(model_size="base")
    except Exception as e:
        st.session_state["_stt_error"] = str(e)
        return None


@st.cache_resource
def get_tts_engine():
    try:
        return TextToSpeech()
    except Exception as e:
        st.session_state["_tts_error"] = str(e)
        return None


# ---- session state setup ------------------------------------------------
if "questions" not in st.session_state:
    st.session_state.questions = []
if "current_q" not in st.session_state:
    st.session_state.current_q = 0
if "evaluations" not in st.session_state:
    st.session_state.evaluations = []
if "pending_transcript" not in st.session_state:
    st.session_state.pending_transcript = ""

if not MIC_AVAILABLE:
    st.info(
        "🎙️ Live microphone recording isn't available in this environment. "
        "You can still upload an audio file to transcribe, or just type your answers.",
        icon="ℹ️",
    )

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
                if tts is None or not tts.available:
                    st.warning(
                        "Text-to-speech isn't available here - here's the question "
                        f"in text instead:\n\n**{clean_question}**"
                    )
                else:
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
            record_clicked = st.button(
                "🎙️ Record answer",
                key=f"record_{idx}",
                disabled=not MIC_AVAILABLE,
            )
            if record_clicked and MIC_AVAILABLE:
                stt = get_stt_engine()
                if stt is None:
                    st.error("Speech-to-text engine failed to load. Please type your answer below.")
                else:
                    audio_path = f"answer_{idx}.wav"
                    try:
                        with st.spinner(f"Recording for {record_seconds}s... speak now"):
                            stt.record_audio(duration=int(record_seconds), filename=audio_path)
                        with st.spinner("Transcribing..."):
                            transcript = stt.transcribe(audio_path)
                        st.session_state.pending_transcript = transcript
                    except Exception as e:
                        st.error(f"Recording failed: {e}. Please type your answer instead.")
                    finally:
                        # clean up the temp audio file
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
                    st.rerun()

        # ---- fallback: upload a pre-recorded audio file to transcribe ----
        # Transcription only needs a file on disk, not a live input device,
        # so this works even when MIC_AVAILABLE is False.
        with st.expander("Or upload an audio file of your answer instead"):
            uploaded_audio = st.file_uploader(
                "Audio file (wav, mp3, m4a)",
                type=["wav", "mp3", "m4a"],
                key=f"audio_upload_{idx}",
            )
            if uploaded_audio is not None and st.button("Transcribe uploaded audio", key=f"transcribe_{idx}"):
                stt = get_stt_engine()
                if stt is None:
                    st.error("Speech-to-text engine failed to load. Please type your answer below.")
                else:
                    audio_path = f"uploaded_answer_{idx}.wav"
                    try:
                        with open(audio_path, "wb") as f:
                            f.write(uploaded_audio.getbuffer())
                        with st.spinner("Transcribing..."):
                            transcript = stt.transcribe(audio_path)
                        st.session_state.pending_transcript = transcript
                    except Exception as e:
                        st.error(f"Transcription failed: {e}. Please type your answer instead.")
                    finally:
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
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