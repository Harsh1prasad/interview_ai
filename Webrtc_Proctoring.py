"""
streamlit-webrtc glue for ProctoringMonitor.

Kept separate from proctoring.py so the core detection logic has no
Streamlit/WebRTC dependency. `webrtc_streamer` runs the video processor on a
background thread per browser connection - detection results are pushed
onto a thread-safe queue.Queue rather than written directly to
st.session_state, because touching session_state from a thread other than
the main script thread is unsafe in Streamlit. The main script drains the
queue on each rerun instead (see drain_events()).
"""
import queue
import threading

from proctoring import ProctoringMonitor, YOLO_AVAILABLE

try:
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
    WEBRTC_ERROR = None
except ImportError as e:
    VideoProcessorBase = object
    webrtc_streamer = None
    RTCConfiguration = None
    av = None
    WEBRTC_AVAILABLE = False
    WEBRTC_ERROR = str(e)
    print(f"[proctoring] streamlit-webrtc not available: {e}")

RTC_CONFIGURATION = (
    RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    if WEBRTC_AVAILABLE else None
)


class ProctoringVideoProcessor(VideoProcessorBase):
    """One instance is created per active browser camera connection."""

    def __init__(self, yolo_model=None):
        self.monitor = ProctoringMonitor(
            enable_object_detection=(yolo_model is not None) or YOLO_AVAILABLE,
            yolo_model=yolo_model,
        )
        self.event_queue = queue.Queue()
        self._lock = threading.Lock()
        self.frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        with self._lock:
            annotated, new_events = self.monitor.process_frame(img)
            self.frame_count += 1
        for ev in new_events:
            self.event_queue.put(ev)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def start_proctoring_stream(key, yolo_model=None):
    """
    Renders the webcam widget + runs live detection. Returns the
    webrtc_streamer context (or None if streamlit-webrtc isn't available).
    """
    if not WEBRTC_AVAILABLE:
        return None

    return webrtc_streamer(
        key=key,
        video_processor_factory=lambda: ProctoringVideoProcessor(yolo_model=yolo_model),
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )


def drain_events(webrtc_ctx):
    """Pull any newly-queued violation events off the active processor's queue."""
    events = []
    if webrtc_ctx is None or webrtc_ctx.video_processor is None:
        return events
    q = webrtc_ctx.video_processor.event_queue
    while True:
        try:
            events.append(q.get_nowait())
        except queue.Empty:
            break
    return events