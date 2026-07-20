import os
import re
import time
import streamlit as st

from question_generation import question_generation
from evaluator import ans_evaluation
from stt_tts.stt import SpeechToText
from stt_tts.tts import TextToSpeech
from Proctoring import YOLO_AVAILABLE
from Webrtc_Proctoring import WEBRTC_AVAILABLE, WEBRTC_ERROR, start_proctoring_stream, drain_events

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


@st.cache_resource
def get_yolo_model():
    """Load the phone/object-detection model once and reuse it across all
    proctoring sessions, instead of re-downloading/loading YOLO weights
    every rerun."""
    if not YOLO_AVAILABLE:
        return None
    try:
        from ultralytics import YOLO
        return YOLO("yolov8n.pt")
    except Exception as e:
        st.session_state["_yolo_error"] = str(e)
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
if "proctoring_events" not in st.session_state:
    st.session_state.proctoring_events = []

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

    # ---- proctoring: live webcam monitoring ------------------------------
    # Uses the BROWSER's camera via WebRTC (same reasoning as audio_input
    # above) - cv2.VideoCapture(0) would try to open a webcam on the
    # server, which doesn't exist when this app is deployed.
    with st.expander("🎥 Proctoring", expanded=True):
        if not WEBRTC_AVAILABLE:
            st.warning(
                "Webcam proctoring isn't available right now "
                f"({WEBRTC_ERROR}). The interview will continue without it."
            )
        else:
            st.caption(
                "Stay visible and facing the screen. Flags: face not visible, "
                "multiple faces, looking away, phone/device detected."
            )
            yolo_model = get_yolo_model()
            webrtc_ctx = start_proctoring_stream(key="proctoring_stream", yolo_model=yolo_model)

            # Drain any violations detected since the last rerun. This runs
            # on every rerun (button clicks, answer submissions, etc.), so
            # the log below won't update instantly frame-by-frame, but it
            # will always be current by the time you submit an answer.
            new_events = drain_events(webrtc_ctx)
            if new_events:
                st.session_state.proctoring_events.extend(new_events)

            if st.session_state.proctoring_events:
                recent = st.session_state.proctoring_events[-5:]
                st.warning(
                    "⚠️ " + " | ".join(
                        f"{e['type'].replace('_', ' ')}" for e in recent
                    )
                )
                st.caption(f"{len(st.session_state.proctoring_events)} flag(s) so far this session.")
            elif webrtc_ctx is not None and webrtc_ctx.state.playing:
                st.success("No proctoring flags yet.")

    idx = st.session_state.current_q
    total = len(st.session_state.questions)

    if idx < total:
        question_text = st.session_state.questions[idx]
        # strip the leading "1. " / "2) " numbering before speaking it aloud
        clean_question = re.sub(r"^\d+[\.\)]\s*", "", question_text)

        st.write(f"**Question {idx + 1} of {total}**")
        st.write(question_text)

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

        # ---- live recording via the BROWSER's microphone -----------------
        # st.audio_input captures audio in the user's browser (HTML5 Media
        # Recorder API) and sends the bytes back to this server. Unlike
        # `sounddevice`, this does NOT need PortAudio or a mic attached to
        # wherever the app is hosted, so it works both locally and deployed
        # (e.g. Streamlit Community Cloud). Requires streamlit >= 1.34.0.
        st.write("🎙️ Record your answer:")
        audio_value = st.audio_input("Record your answer", key=f"audio_input_{idx}", label_visibility="collapsed")

        if audio_value is not None:
            if st.button("Transcribe recording", key=f"transcribe_live_{idx}"):
                stt = get_stt_engine()
                if stt is None:
                    st.error("Speech-to-text engine failed to load. Please type your answer below.")
                else:
                    audio_path = f"answer_{idx}.wav"
                    try:
                        with open(audio_path, "wb") as f:
                            f.write(audio_value.getbuffer())
                        with st.spinner("Transcribing..."):
                            transcript = stt.transcribe(audio_path)
                        st.session_state.pending_transcript = transcript
                    except Exception as e:
                        st.error(f"Transcription failed: {e}. Please type your answer instead.")
                    finally:
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
                    st.rerun()

        # ---- fallback: upload a pre-recorded audio file to transcribe ----
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

    if st.session_state.proctoring_events:
        st.subheader("Proctoring report")
        counts = {}
        for ev in st.session_state.proctoring_events:
            counts[ev["type"]] = counts.get(ev["type"], 0) + 1
        for vtype, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            st.write(f"- **{vtype.replace('_', ' ').title()}**: {count} time(s)")
        with st.expander("Full proctoring timeline"):
            for ev in st.session_state.proctoring_events:
                st.write(f"`{time.strftime('%H:%M:%S', time.localtime(ev['timestamp']))}` — {ev['message']}")

    for i, e in enumerate(st.session_state.evaluations, start=1):
        with st.expander(f"Q{i}: {e['question']}"):
            st.write(f"**Your answer:** {e['answer']}")
            st.write("**Feedback:**")
            st.write(e["feedback"])




# import os
# import re
# import time
# import uuid
# import streamlit as st

# from question_generation import question_generation
# from evaluator import ans_evaluation
# from stt_tts.stt import SpeechToText
# from stt_tts.tts import TextToSpeech
# from Proctoring import YOLO_AVAILABLE
# from Webrtc_Proctoring import WEBRTC_AVAILABLE, WEBRTC_ERROR, start_proctoring_stream, drain_events

# st.set_page_config(page_title="AI Interview Prep", page_icon="▚", layout="centered")

# # ============================================================================
# # THEME — "diagnostics console" look: dark, monospace, panel/log styling.
# # Everything below the imports is presentation only; app logic/state/keys
# # are unchanged from the original implementation.
# # ============================================================================

# if "session_id" not in st.session_state:
#     st.session_state.session_id = uuid.uuid4().hex[:10]

# CSS = """
# <style>
# @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

# :root{
#   --bg:        #0a0d10;
#   --panel:     #10151b;
#   --panel-alt: #131a21;
#   --line:      #212a33;
#   --line-soft: #182029;
#   --text:      #d9e2ea;
#   --dim:       #6c7986;
#   --accent:    #35e28a;   /* terminal green   */
#   --accent-2:  #f5a623;   /* amber / warn      */
#   --accent-3:  #ff5f57;   /* red / flag        */
#   --accent-4:  #4fa3ff;   /* blue / info       */
#   --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
# }

# html, body, [class*="css"]{ font-family: var(--mono) !important; }

# .stApp{
#   background:
#     repeating-linear-gradient(0deg, rgba(255,255,255,0.012) 0px, rgba(255,255,255,0.012) 1px, transparent 1px, transparent 3px),
#     radial-gradient(circle at 15% 0%, #0d1319 0%, var(--bg) 45%);
#   color: var(--text);
# }

# /* kill default streamlit chrome noise */
# #MainMenu, footer, header{ visibility: hidden; }
# .block-container{ padding-top: 1.6rem; max-width: 760px; }

# /* ---------- terminal title bar (signature element) ---------- */
# .term-bar{
#   border: 1px solid var(--line);
#   border-bottom: none;
#   border-radius: 8px 8px 0 0;
#   background: linear-gradient(180deg, #141b22, #10151b);
#   padding: 9px 14px;
#   display: flex; align-items: center; gap: 10px;
# }
# .term-dot{ width: 10px; height: 10px; border-radius: 50%; display:inline-block; }
# .term-dot.r{ background:#ff5f57; } .term-dot.y{ background:#febc2e; } .term-dot.g{ background:#28c840; }
# .term-path{ margin-left: 6px; color: var(--dim); font-size: 12px; letter-spacing: .03em; }
# .term-body{
#   border: 1px solid var(--line);
#   border-radius: 0 0 8px 8px;
#   background: var(--panel);
#   padding: 22px 24px 20px 24px;
#   margin-bottom: 22px;
# }
# .term-title{
#   font-size: 22px; font-weight: 800; letter-spacing: .01em; color: var(--text);
#   margin: 0 0 4px 0;
# }
# .term-title .cursor{
#   display:inline-block; width:9px; height:19px; background: var(--accent);
#   margin-left: 6px; vertical-align: -3px;
#   animation: blink 1.05s steps(1) infinite;
# }
# @keyframes blink{ 50%{ opacity: 0; } }
# .term-sub{ color: var(--dim); font-size: 12.5px; }
# .term-sub b{ color: var(--accent); font-weight: 600; }

# /* ---------- pipeline / stage strip ---------- */
# .stage-strip{ display:flex; gap:6px; margin: 4px 0 26px 0; }
# .stage{
#   flex:1; text-align:center; font-size: 11px; letter-spacing:.06em;
#   padding: 7px 4px; border: 1px solid var(--line); border-radius: 5px;
#   color: var(--dim); background: var(--panel-alt);
# }
# .stage.active{ color:#04150c; background: var(--accent); border-color: var(--accent); font-weight:700; }
# .stage.done{ color: var(--accent); border-color: #1c3d2c; }

# /* ---------- generic technical panel ---------- */
# .panel{
#   border: 1px solid var(--line); border-radius: 7px; background: var(--panel);
#   padding: 16px 18px; margin-bottom: 16px;
# }
# .panel-head{
#   display:flex; justify-content: space-between; align-items:center;
#   font-size: 11px; letter-spacing:.08em; color: var(--dim); margin-bottom: 10px;
#   text-transform: uppercase; border-bottom: 1px dashed var(--line-soft); padding-bottom: 8px;
# }
# .panel-head .tag{ color: var(--accent); }
# .qid-badge{
#   display:inline-block; font-size: 11px; font-weight:700; letter-spacing:.04em;
#   color: #04150c; background: var(--accent); padding: 2px 8px; border-radius: 4px; margin-right: 8px;
# }
# .q-text{ font-size: 15px; line-height: 1.55; color: var(--text); }

# /* ---------- log lines (proctoring) ---------- */
# .logline{
#   font-size: 12.5px; padding: 3px 0; color: var(--text); border-bottom: 1px dotted var(--line-soft);
# }
# .logline .ts{ color: var(--dim); }
# .logline .sev-warn{ color: var(--accent-2); font-weight: 700; }
# .logline .sev-crit{ color: var(--accent-3); font-weight: 700; }

# /* ---------- streamlit widget overrides ---------- */
# h1,h2,h3{ color: var(--text) !important; font-family: var(--mono) !important; letter-spacing: .01em; }
# h2{ font-size: 16px !important; text-transform: uppercase; letter-spacing: .08em; color: var(--dim) !important; border-left: 3px solid var(--accent); padding-left: 10px; }
# h3{ font-size: 14px !important; }

# p, span, label, .stMarkdown{ color: var(--text); }

# .stButton > button{
#   font-family: var(--mono) !important; font-weight: 700; font-size: 12.5px;
#   letter-spacing: .04em; text-transform: uppercase;
#   background: var(--panel-alt); color: var(--accent);
#   border: 1px solid #1c3d2c; border-radius: 5px; padding: 8px 16px;
#   transition: all .12s ease;
# }
# .stButton > button:hover{ background: var(--accent); color: #04150c; border-color: var(--accent); }
# .stButton > button:active{ transform: translateY(1px); }

# [data-testid="stFileUploader"]{
#   border: 1px dashed var(--line); border-radius: 7px; padding: 6px; background: var(--panel-alt);
# }
# [data-testid="stFileUploaderDropzone"]{ background: transparent !important; }

# .stTextArea textarea, .stTextInput input{
#   font-family: var(--mono) !important; background: #0d1218 !important; color: var(--text) !important;
#   border: 1px solid var(--line) !important; border-radius: 6px !important;
# }
# .stTextArea textarea:focus, .stTextInput input:focus{ border-color: var(--accent) !important; box-shadow: 0 0 0 1px var(--accent) !important; }

# [data-testid="stExpander"]{
#   border: 1px solid var(--line) !important; border-radius: 7px !important; background: var(--panel) !important;
#   overflow: hidden;
# }
# [data-testid="stExpander"] summary{ font-family: var(--mono) !important; font-size: 12.5px !important; }

# div[data-testid="stAlert"]{ font-family: var(--mono) !important; border-radius: 6px !important; font-size: 13px !important; }

# [data-testid="stSidebar"]{
#   background: #0c1015; border-right: 1px solid var(--line);
# }
# [data-testid="stSidebar"] *{ font-family: var(--mono) !important; }

# .stProgress > div > div{ background-color: var(--accent) !important; }

# hr{ border-color: var(--line) !important; }

# /* audio widgets / spinner text */
# .stSpinner > div{ color: var(--accent) !important; }
# </style>
# """
# st.markdown(CSS, unsafe_allow_html=True)


# def render_header(stage_index: int) -> None:
#     """Terminal-style masthead + pipeline stage strip."""
#     st.markdown(
#         f"""
#         <div class="term-bar">
#             <span class="term-dot r"></span><span class="term-dot y"></span><span class="term-dot g"></span>
#             <span class="term-path">session::{st.session_state.session_id} — zsh — 100x28</span>
#         </div>
#         <div class="term-body">
#             <div class="term-title">AI_INTERVIEW_PREP<span class="cursor"></span></div>
#             <div class="term-sub">status: <b>ONLINE</b> · engines lazy-loaded on first use · resume/JD parsed locally</div>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

#     stages = ["01_UPLOAD", "02_INTERVIEW", "03_REPORT"]
#     chips = []
#     for i, label in enumerate(stages):
#         cls = "stage"
#         if i < stage_index:
#             cls += " done"
#         elif i == stage_index:
#             cls += " active"
#         chips.append(f'<div class="{cls}">{label}</div>')
#     st.markdown(f'<div class="stage-strip">{"".join(chips)}</div>', unsafe_allow_html=True)


# # ---- cached engines (loaded once, reused across reruns) ------------------
# # NOTE: SpeechToText() loading a Whisper model and TextToSpeech() trying to
# # init a system engine can both still fail for reasons other than "no mic"
# # (e.g. out of memory, no espeak binary). Wrap in try/except too, so a
# # surprise failure degrades the feature instead of crashing the app.
# @st.cache_resource
# def get_stt_engine():
#     try:
#         return SpeechToText(model_size="base")
#     except Exception as e:
#         st.session_state["_stt_error"] = str(e)
#         return None


# @st.cache_resource
# def get_tts_engine():
#     try:
#         return TextToSpeech()
#     except Exception as e:
#         st.session_state["_tts_error"] = str(e)
#         return None


# @st.cache_resource
# def get_yolo_model():
#     """Load the phone/object-detection model once and reuse it across all
#     proctoring sessions, instead of re-downloading/loading YOLO weights
#     every rerun."""
#     if not YOLO_AVAILABLE:
#         return None
#     try:
#         from ultralytics import YOLO
#         return YOLO("yolov8n.pt")
#     except Exception as e:
#         st.session_state["_yolo_error"] = str(e)
#         return None


# # ---- session state setup ------------------------------------------------
# if "questions" not in st.session_state:
#     st.session_state.questions = []
# if "current_q" not in st.session_state:
#     st.session_state.current_q = 0
# if "evaluations" not in st.session_state:
#     st.session_state.evaluations = []
# if "pending_transcript" not in st.session_state:
#     st.session_state.pending_transcript = ""
# if "proctoring_events" not in st.session_state:
#     st.session_state.proctoring_events = []

# # ---- sidebar: system status panel ----------------------------------------
# with st.sidebar:
#     st.markdown(
#         '<div class="panel-head" style="border:none;margin-bottom:2px;"><span>&gt; system_status</span></div>',
#         unsafe_allow_html=True,
#     )
#     st.markdown(f"<div class='logline'><span class='ts'>session</span> {st.session_state.session_id}</div>", unsafe_allow_html=True)
#     st.markdown(
#         "<div class='logline'><span class='ts'>stt/tts</span> lazy "
#         "<span class='sev-warn'>[loads on first use]</span></div>",
#         unsafe_allow_html=True,
#     )
#     yolo_badge = '<span style="color:var(--accent)">available</span>' if YOLO_AVAILABLE else '<span class="sev-warn">unavailable</span>'
#     st.markdown(f"<div class='logline'><span class='ts'>yolo</span> {yolo_badge}</div>", unsafe_allow_html=True)
#     webrtc_badge = '<span style="color:var(--accent)">available</span>' if WEBRTC_AVAILABLE else '<span class="sev-warn">unavailable</span>'
#     st.markdown(f"<div class='logline'><span class='ts'>webrtc</span> {webrtc_badge}</div>", unsafe_allow_html=True)
#     total_q = len(st.session_state.questions)
#     answered = len(st.session_state.evaluations)
#     st.markdown(
#         f"<div class='logline'><span class='ts'>progress</span> {answered}/{total_q or '-'}</div>",
#         unsafe_allow_html=True,
#     )
#     flag_count = len(st.session_state.proctoring_events)
#     sev = "sev-crit" if flag_count else ""
#     st.markdown(
#         f"<div class='logline'><span class='ts'>flags</span> <span class='{sev}'>{flag_count}</span></div>",
#         unsafe_allow_html=True,
#     )

# # ---- header + pipeline stage indicator ------------------------------------
# if st.session_state.evaluations and st.session_state.current_q >= len(st.session_state.questions) and st.session_state.questions:
#     _stage = 2
# elif st.session_state.questions:
#     _stage = 1
# else:
#     _stage = 0
# render_header(_stage)

# # ---- step 1: upload resume + JD, generate questions ----------------------
# st.markdown('<h2>01 · Upload</h2>', unsafe_allow_html=True)
# with st.container(border=True):
#     st.markdown(
#         '<div class="panel-head"><span>&gt; input_documents</span><span class="tag">PDF</span></div>',
#         unsafe_allow_html=True,
#     )
#     resume = st.file_uploader("resume.pdf", type=["pdf"])
#     jd = st.file_uploader("job_description.pdf", type=["pdf"])

#     if resume and jd and st.button("▶ Generate questions"):
#         with st.spinner("Generating interview questions..."):
#             raw_questions = question_generation(resume, jd)
#             questions = [q.strip() for q in raw_questions if re.match(r"^\d+[\.\)]", q.strip())]

#         if not questions:
#             st.error("Couldn't parse any questions from the model's response. Please try again.")
#         else:
#             st.session_state.questions = questions
#             st.session_state.current_q = 0
#             st.session_state.evaluations = []
#             st.rerun()

# # ---- step 2: walk through questions, collect + evaluate answers ----------
# if st.session_state.questions:
#     st.markdown('<h2>02 · Interview</h2>', unsafe_allow_html=True)

#     # ---- proctoring: live webcam monitoring ------------------------------
#     # Uses the BROWSER's camera via WebRTC (same reasoning as audio_input
#     # above) - cv2.VideoCapture(0) would try to open a webcam on the
#     # server, which doesn't exist when this app is deployed.
#     with st.expander("◉ Proctoring feed", expanded=True):
#         if not WEBRTC_AVAILABLE:
#             st.warning(
#                 "Webcam proctoring isn't available right now "
#                 f"({WEBRTC_ERROR}). The interview will continue without it."
#             )
#         else:
#             st.caption(
#                 "Stay visible and facing the screen. Flags: face not visible, "
#                 "multiple faces, looking away, phone/device detected."
#             )
#             yolo_model = get_yolo_model()
#             webrtc_ctx = start_proctoring_stream(key="proctoring_stream", yolo_model=yolo_model)

#             # Drain any violations detected since the last rerun. This runs
#             # on every rerun (button clicks, answer submissions, etc.), so
#             # the log below won't update instantly frame-by-frame, but it
#             # will always be current by the time you submit an answer.
#             new_events = drain_events(webrtc_ctx)
#             if new_events:
#                 st.session_state.proctoring_events.extend(new_events)

#             if st.session_state.proctoring_events:
#                 recent = st.session_state.proctoring_events[-5:]
#                 lines = []
#                 for e in recent:
#                     ts = time.strftime('%H:%M:%S', time.localtime(e.get('timestamp', time.time())))
#                     sev_cls = "sev-crit" if "phone" in e["type"] else "sev-warn"
#                     lines.append(
#                         f"<div class='logline'><span class='ts'>[{ts}]</span> "
#                         f"<span class='{sev_cls}'>{e['type'].replace('_', ' ')}</span></div>"
#                     )
#                 log_html = "".join(lines)
#                 st.markdown(
#                     f"<div class='panel' style='margin-top:8px;'><div class='panel-head'>"
#                     f"<span>&gt; violation_log</span><span class='tag'>last {len(recent)}</span></div>"
#                     f"{log_html}</div>",
#                     unsafe_allow_html=True,
#                 )
#                 st.caption(f"{len(st.session_state.proctoring_events)} flag(s) so far this session.")
#             elif webrtc_ctx is not None and webrtc_ctx.state.playing:
#                 st.success("No proctoring flags yet.")

#     idx = st.session_state.current_q
#     total = len(st.session_state.questions)

#     if idx < total:
#         question_text = st.session_state.questions[idx]
#         # strip the leading "1. " / "2) " numbering before speaking it aloud
#         clean_question = re.sub(r"^\d+[\.\)]\s*", "", question_text)

#         with st.container(border=True):
#             st.markdown(
#                 f"""
#                 <div class="panel-head"><span>&gt; current_question</span><span class="tag">{idx + 1}/{total}</span></div>
#                 <div><span class="qid-badge">Q_{idx + 1:02d}</span><span class="q-text">{clean_question}</span></div>
#                 """,
#                 unsafe_allow_html=True,
#             )

#             if st.button("🔊 Read aloud", key=f"tts_{idx}"):
#                 tts = get_tts_engine()
#                 if tts is None or not tts.available:
#                     st.warning(
#                         "Text-to-speech isn't available here - here's the question "
#                         f"in text instead:\n\n**{clean_question}**"
#                     )
#                 else:
#                     with st.spinner("Speaking..."):
#                         tts.speak(clean_question)

#         # ---- live recording via the BROWSER's microphone -----------------
#         # st.audio_input captures audio in the user's browser (HTML5 Media
#         # Recorder API) and sends the bytes back to this server. Unlike
#         # `sounddevice`, this does NOT need PortAudio or a mic attached to
#         # wherever the app is hosted, so it works both locally and deployed
#         # (e.g. Streamlit Community Cloud). Requires streamlit >= 1.34.0.
#         st.markdown('<h2>Record answer</h2>', unsafe_allow_html=True)
#         with st.container(border=True):
#             st.markdown('<div class="panel-head"><span>&gt; mic_input</span></div>', unsafe_allow_html=True)
#             audio_value = st.audio_input("Record your answer", key=f"audio_input_{idx}", label_visibility="collapsed")

#             if audio_value is not None:
#                 if st.button("⏵ Transcribe recording", key=f"transcribe_live_{idx}"):
#                     stt = get_stt_engine()
#                     if stt is None:
#                         st.error("Speech-to-text engine failed to load. Please type your answer below.")
#                     else:
#                         audio_path = f"answer_{idx}.wav"
#                         try:
#                             with open(audio_path, "wb") as f:
#                                 f.write(audio_value.getbuffer())
#                             with st.spinner("Transcribing..."):
#                                 transcript = stt.transcribe(audio_path)
#                             st.session_state.pending_transcript = transcript
#                         except Exception as e:
#                             st.error(f"Transcription failed: {e}. Please type your answer instead.")
#                         finally:
#                             if os.path.exists(audio_path):
#                                 os.remove(audio_path)
#                         st.rerun()

#             # ---- fallback: upload a pre-recorded audio file to transcribe ----
#             with st.expander("↳ Upload an audio file instead"):
#                 uploaded_audio = st.file_uploader(
#                     "Audio file (wav, mp3, m4a)",
#                     type=["wav", "mp3", "m4a"],
#                     key=f"audio_upload_{idx}",
#                 )
#                 if uploaded_audio is not None and st.button("⏵ Transcribe uploaded audio", key=f"transcribe_{idx}"):
#                     stt = get_stt_engine()
#                     if stt is None:
#                         st.error("Speech-to-text engine failed to load. Please type your answer below.")
#                     else:
#                         audio_path = f"uploaded_answer_{idx}.wav"
#                         try:
#                             with open(audio_path, "wb") as f:
#                                 f.write(uploaded_audio.getbuffer())
#                             with st.spinner("Transcribing..."):
#                                 transcript = stt.transcribe(audio_path)
#                             st.session_state.pending_transcript = transcript
#                         except Exception as e:
#                             st.error(f"Transcription failed: {e}. Please type your answer instead.")
#                         finally:
#                             if os.path.exists(audio_path):
#                                 os.remove(audio_path)
#                         st.rerun()

#         answer_key = f"answer_{idx}"
#         # if a recording was just transcribed, seed the text area with it
#         # (must be set BEFORE the widget is instantiated)
#         if st.session_state.pending_transcript:
#             st.session_state[answer_key] = st.session_state.pending_transcript
#             st.session_state.pending_transcript = ""

#         st.markdown('<h2>Write / edit answer</h2>', unsafe_allow_html=True)
#         answer = st.text_area("Your answer", key=answer_key, label_visibility="collapsed", height=140)

#         if st.button("■ Submit answer", key=f"submit_{idx}"):
#             if answer.strip():
#                 with st.spinner("Evaluating your answer..."):
#                     feedback = ans_evaluation(question_text, answer)
#                 st.session_state.evaluations.append(
#                     {
#                         "question": question_text,
#                         "answer": answer,
#                         "feedback": feedback,
#                     }
#                 )
#                 st.session_state.current_q += 1
#                 st.rerun()
#             else:
#                 st.warning("Please enter an answer before submitting.")
#     else:
#         st.success("Interview complete! See your feedback below.")

# # ---- step 3: feedback summary --------------------------------------------
# if st.session_state.evaluations:
#     st.markdown('<h2>03 · Report</h2>', unsafe_allow_html=True)

#     if st.session_state.proctoring_events:
#         st.markdown('<h3>Proctoring report</h3>', unsafe_allow_html=True)
#         counts = {}
#         for ev in st.session_state.proctoring_events:
#             counts[ev["type"]] = counts.get(ev["type"], 0) + 1
#         rows = "".join(
#             f"<div class='logline'><span class='ts'>{vtype.replace('_', ' ').upper()}</span> "
#             f"<span class='sev-warn'>× {count}</span></div>"
#             for vtype, count in sorted(counts.items(), key=lambda kv: -kv[1])
#         )
#         st.markdown(f"<div class='panel'>{rows}</div>", unsafe_allow_html=True)
#         with st.expander("Full proctoring timeline"):
#             timeline_html = "".join(
#                 f"<div class='logline'><span class='ts'>[{time.strftime('%H:%M:%S', time.localtime(ev['timestamp']))}]</span> {ev['message']}</div>"
#                 for ev in st.session_state.proctoring_events
#             )
#             st.markdown(timeline_html, unsafe_allow_html=True)

#     for i, e in enumerate(st.session_state.evaluations, start=1):
#         with st.expander(f"Q_{i:02d} · {e['question']}"):
#             st.markdown(
#                 "<div class='panel-head'><span>&gt; your_answer</span></div>"
#                 f"<div class='q-text'>{e['answer']}</div>",
#                 unsafe_allow_html=True,
#             )
#             st.markdown(
#                 "<div class='panel-head' style='margin-top:14px;'><span>&gt; feedback</span></div>",
#                 unsafe_allow_html=True,
#             )
#             st.write(e["feedback"])




