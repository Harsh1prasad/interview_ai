"""
Core proctoring detection logic.

Deliberately has no Streamlit/WebRTC dependency - it just takes BGR frames
(numpy arrays) in and returns annotated frames + violation events out. This
makes it easy to unit test or reuse from a different UI later.
"""
import time

import numpy as np

try:
    import cv2
    import mediapipe as mp
    CV_AVAILABLE = True
    CV_ERROR = None
except ImportError as e:
    cv2 = None
    mp = None
    CV_AVAILABLE = False
    CV_ERROR = str(e)
    print(f"[proctoring] opencv/mediapipe not available: {e}")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    YOLO_ERROR = None
except ImportError as e:
    YOLO = None
    YOLO_AVAILABLE = False
    YOLO_ERROR = str(e)
    print(f"[proctoring] ultralytics not available: {e}")


# COCO class name for a phone is "cell phone". A couple of other classes are
# included since a candidate could be using a tablet/second device off to
# the side rather than a phone specifically.
DEVICE_CLASS_NAMES = {"cell phone", "laptop", "tablet", "remote"}

# A condition must persist for this many consecutive seconds before it's
# logged as a violation - avoids flagging a single blink / camera glitch.
NO_FACE_GRACE_SECONDS = 3.0
LOOK_AWAY_GRACE_SECONDS = 2.5

# Don't log the same violation type again within this many seconds, so a
# sustained issue (e.g. candidate stays out of frame) produces one event
# every few seconds instead of one per video frame.
VIOLATION_COOLDOWN_SECONDS = 5.0

# Head-pose angle (degrees) beyond which we call it "looking away". These
# are rough heuristics, not a calibrated biometric measurement - tune them
# if you find false positives/negatives in practice.
YAW_THRESHOLD_DEGREES = 25.0
PITCH_THRESHOLD_DEGREES = 20.0

# Generic 3D face landmark positions (mm), used for solvePnP head-pose
# estimation. Indices match MediaPipe FaceMesh landmark ids.
_FACE_MESH_IDX = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye": 33,
    "right_eye": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}
_MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),        # nose tip
    (0.0, -63.6, -12.5),    # chin
    (-43.3, 32.7, -26.0),   # left eye corner
    (43.3, 32.7, -26.0),    # right eye corner
    (-28.9, -28.9, -24.1),  # left mouth corner
    (28.9, -28.9, -24.1),   # right mouth corner
])


class ProctoringMonitor:
    """
    Stateful, frame-by-frame proctoring monitor.

    Usage:
        monitor = ProctoringMonitor()
        annotated_frame, new_events = monitor.process_frame(bgr_frame)
        ...
        report = monitor.summary()

    Pass a pre-loaded `yolo_model` (e.g. loaded once via st.cache_resource)
    to avoid reloading YOLO weights on every instantiation.
    """

    def __init__(self, enable_object_detection=True, target_classes=None, yolo_model=None):
        self.available = CV_AVAILABLE
        self.object_detection_available = False
        self.violations = []       # full history: list of event dicts
        self._last_logged = {}     # violation_type -> last time logged
        self._no_face_since = None
        self._look_away_since = None
        self._yolo = None

        if not self.available:
            return

        self._mp_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.target_classes = target_classes or DEVICE_CLASS_NAMES

        if enable_object_detection:
            if yolo_model is not None:
                self._yolo = yolo_model
                self.object_detection_available = True
            elif YOLO_AVAILABLE:
                try:
                    self._yolo = YOLO("yolov8n.pt")
                    self.object_detection_available = True
                except Exception as e:
                    print(f"[proctoring] Could not load YOLO model, object detection disabled: {e}")
                    self._yolo = None

    def _log(self, violation_type, message):
        now = time.time()
        last = self._last_logged.get(violation_type, 0)
        if now - last < VIOLATION_COOLDOWN_SECONDS:
            return None
        self._last_logged[violation_type] = now
        event = {"type": violation_type, "message": message, "timestamp": now}
        self.violations.append(event)
        return event

    def _estimate_head_pose(self, landmarks, frame_w, frame_h):
        """Rough yaw/pitch (degrees) via solvePnP against a generic 3D face model."""
        try:
            image_points = np.array([
                (landmarks[_FACE_MESH_IDX[name]].x * frame_w, landmarks[_FACE_MESH_IDX[name]].y * frame_h)
                for name in ("nose_tip", "chin", "left_eye", "right_eye", "left_mouth", "right_mouth")
            ], dtype="double")
        except IndexError:
            return None, None

        focal_length = frame_w
        center = (frame_w / 2, frame_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype="double")
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, _ = cv2.solvePnP(
            _MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return None, None

        rmat, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        yaw = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
        return yaw, pitch

    def process_frame(self, frame_bgr):
        """
        frame_bgr: numpy array, BGR (as from OpenCV / streamlit-webrtc).
        Returns (annotated_frame_bgr, new_violations: list[dict]).
        """
        if not self.available:
            return frame_bgr, []

        new_events = []
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        annotated = frame_bgr.copy()
        now = time.time()

        # ---- face presence / count ----
        det_result = self._mp_face_detection.process(rgb)
        num_faces = len(det_result.detections) if det_result.detections else 0

        if det_result.detections:
            for det in det_result.detections:
                bbox = det.location_data.relative_bounding_box
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                bw, bh = int(bbox.width * w), int(bbox.height * h)
                cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 200, 0), 2)

        if num_faces == 0:
            if self._no_face_since is None:
                self._no_face_since = now
            elif now - self._no_face_since >= NO_FACE_GRACE_SECONDS:
                ev = self._log("no_face", "No face detected - candidate may have left the frame.")
                if ev:
                    new_events.append(ev)
        else:
            self._no_face_since = None

        if num_faces > 1:
            ev = self._log("multiple_faces", f"{num_faces} faces detected in frame.")
            if ev:
                new_events.append(ev)

        # ---- gaze / head pose (only meaningful with exactly one face) ----
        if num_faces == 1:
            mesh_result = self._mp_face_mesh.process(rgb)
            if mesh_result.multi_face_landmarks:
                landmarks = mesh_result.multi_face_landmarks[0].landmark
                yaw, pitch = self._estimate_head_pose(landmarks, w, h)
                if yaw is not None:
                    looking_away = abs(yaw) > YAW_THRESHOLD_DEGREES or abs(pitch) > PITCH_THRESHOLD_DEGREES
                    if looking_away:
                        if self._look_away_since is None:
                            self._look_away_since = now
                        elif now - self._look_away_since >= LOOK_AWAY_GRACE_SECONDS:
                            ev = self._log("looking_away", "Candidate appears to be looking away from the screen.")
                            if ev:
                                new_events.append(ev)
                    else:
                        self._look_away_since = None
        else:
            self._look_away_since = None

        # ---- object detection (phone / other devices) ----
        if self._yolo is not None:
            try:
                results = self._yolo.predict(source=frame_bgr, verbose=False, conf=0.45)[0]
                names = results.names
                for box in results.boxes:
                    cls_name = names[int(box.cls[0])]
                    if cls_name in self.target_classes:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
                        cv2.putText(
                            annotated, cls_name, (xyxy[0], max(xyxy[1] - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                        )
                        ev = self._log(f"object_{cls_name}", f"Detected a {cls_name} in frame.")
                        if ev:
                            new_events.append(ev)
            except Exception as e:
                print(f"[proctoring] object detection failed on a frame: {e}")

        return annotated, new_events

    def summary(self):
        """Counts per violation type + full timeline - for the final report."""
        counts = {}
        for v in self.violations:
            counts[v["type"]] = counts.get(v["type"], 0) + 1
        return {"counts": counts, "timeline": self.violations, "total": len(self.violations)}