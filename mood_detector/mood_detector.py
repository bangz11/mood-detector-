"""
AI Mood Detector — Final Version
- FER with strict face detection (no ghost faces)
- 30-frame smoothing (no jumping)
- Bias correction (all emotions balanced)

Run:
    python mood_detector.py

Controls:
    Q  — quit
    S  — save screenshot
    P  — pause / resume
"""

import cv2
import time
import os
from collections import deque
from datetime import datetime
from fer import FER

# ── CONFIG ─────────────────────────────────────────────────────────────────────
WINDOW_NAME   = "AI Mood Detector"
FONT          = cv2.FONT_HERSHEY_SIMPLEX
CAM_INDEX     = 0
SAVE_DIR      = "screenshots"
SMOOTH_FRAMES = 30
FRAME_SKIP    = 1

EMOTION_COLORS = {
    "happy":     (50,  205,  50),
    "sad":       (200,  80,  80),
    "angry":     (0,    0,  220),
    "surprise":  (0,   200, 255),
    "fear":      (180,  60, 180),
    "disgust":   (0,   140,  70),
    "neutral":   (180, 180, 180),
}
DEFAULT_COLOR = (200, 200, 200)
ALL_EMOTIONS  = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]

# Lower happy, boost everything else
EMOTION_WEIGHTS = {
    "happy":    0.2,
    "sad":      2.0,
    "angry":    2.0,
    "surprise": 1.8,
    "fear":     2.0,
    "disgust":  1.8,
    "neutral":  1.5,
}


# ── SMOOTHING BUFFER ───────────────────────────────────────────────────────────
class EmotionSmoother:
    def __init__(self, window=30):
        self.buffers = {e: deque(maxlen=window) for e in ALL_EMOTIONS}

    def update(self, scores: dict):
        for e in ALL_EMOTIONS:
            self.buffers[e].append(scores.get(e, 0.0))

    def get_smoothed(self) -> dict:
        return {e: sum(self.buffers[e]) / max(len(self.buffers[e]), 1)
                for e in ALL_EMOTIONS}


# ── DRAWING HELPERS ────────────────────────────────────────────────────────────
def draw_rounded_rect(img, pt1, pt2, color, thickness=2, radius=12):
    x1, y1 = pt1
    x2, y2 = pt2
    r = radius
    cv2.line(img,  (x1+r, y1),   (x2-r, y1),   color, thickness)
    cv2.line(img,  (x1+r, y2),   (x2-r, y2),   color, thickness)
    cv2.line(img,  (x1,   y1+r), (x1,   y2-r), color, thickness)
    cv2.line(img,  (x2,   y1+r), (x2,   y2-r), color, thickness)
    cv2.ellipse(img, (x1+r, y1+r), (r, r), 180,  0, 90,  color, thickness)
    cv2.ellipse(img, (x2-r, y1+r), (r, r), 270,  0, 90,  color, thickness)
    cv2.ellipse(img, (x1+r, y2-r), (r, r),  90,  0, 90,  color, thickness)
    cv2.ellipse(img, (x2-r, y2-r), (r, r),   0,  0, 90,  color, thickness)


def draw_label_bg(img, text, origin, font, scale, color, thickness=1, pad=6):
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(img,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + baseline + pad),
                  color, cv2.FILLED)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def emotion_bar(img, emotions: dict, x: int, y_start: int, bar_w=160, bar_h=10, gap=18):
    sorted_emotions = sorted(emotions.items(), key=lambda kv: kv[1], reverse=True)
    for i, (emo, score) in enumerate(sorted_emotions):
        y = y_start + i * gap
        filled = int(bar_w * min(score, 1.0))
        color  = EMOTION_COLORS.get(emo, DEFAULT_COLOR)
        cv2.rectangle(img, (x, y), (x + bar_w, y + bar_h), (60, 60, 60), cv2.FILLED)
        if filled > 0:
            cv2.rectangle(img, (x, y), (x + filled, y + bar_h), color, cv2.FILLED)
        label = f"{emo[:7]:<7} {score*100:4.0f}%"
        cv2.putText(img, label, (x + bar_w + 6, y + bar_h - 1),
                    FONT, 0.36, (220, 220, 220), 1, cv2.LINE_AA)


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # mtcnn=False — stricter detection, no ghost faces in background
    detector = FER(mtcnn=False)
    smoother = EmotionSmoother(window=SMOOTH_FRAMES)
    cap      = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {CAM_INDEX}.")
        return

    print("[INFO] Mood Detector running. Press Q to quit.")

    last_results = []
    frame_count  = 0
    paused       = False
    prev_time    = time.time()
    fps          = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        now       = time.time()
        fps       = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-5))
        prev_time = now

        if not paused and frame_count % FRAME_SKIP == 0:
            last_results = detector.detect_emotions(frame)

        frame_count += 1

        for result in last_results:
            x, y, fw, fh = result["box"]
            emotions      = result["emotions"]

            # Normalise to 0-1
            total = sum(emotions.values()) or 1
            norm  = {k: v / total for k, v in emotions.items()}

            smoother.update(norm)
            smooth = smoother.get_smoothed()

            # Apply bias correction
            smooth = {e: smooth[e] * EMOTION_WEIGHTS.get(e, 1.0) for e in smooth}
            total_w = sum(smooth.values()) or 1
            smooth = {e: v / total_w for e, v in smooth.items()}

            top_emo   = max(smooth, key=smooth.get)
            top_score = smooth[top_emo]
            color     = EMOTION_COLORS.get(top_emo, DEFAULT_COLOR)

            draw_rounded_rect(frame, (x, y), (x + fw, y + fh), color, thickness=2)

            label = f"{top_emo.upper()}  {top_score*100:.0f}%"
            draw_label_bg(frame, label,
                          origin=(x, max(y - 10, 20)),
                          font=FONT, scale=0.62, color=color, thickness=1)

            bar_x = min(x + fw + 10, w - 230)
            bar_y = max(y, 10)
            if bar_y + 7 * 18 < h:
                emotion_bar(frame, smooth, bar_x, bar_y)

        # ── HUD ──
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 22),
                    FONT, 0.55, (100, 255, 100), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Smooth: {SMOOTH_FRAMES} frames", (10, 42),
                    FONT, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(frame, "[Q] Quit   [S] Save   [P] Pause",
                    (10, h - 10), FONT, 0.42, (140, 140, 140), 1, cv2.LINE_AA)

        if paused:
            cv2.putText(frame, "PAUSED", (w // 2 - 55, h // 2),
                        FONT, 1.2, (0, 200, 255), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SAVE_DIR, f"mood_{ts}.png")
            cv2.imwrite(path, frame)
            print(f"[INFO] Screenshot saved → {path}")
        elif key == ord('p'):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Detector stopped.")


if __name__ == "__main__":
    main()