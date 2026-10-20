import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision.gesture_recognizer_result import (
    GestureRecognizerResult,
)

model_path = "models/gesture_recognizer.task"
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
RunningMode = mp.tasks.vision.RunningMode
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
)

def display_result(frame, result: GestureRecognizerResult, fps: float) -> None:
    annotated_frame = frame.copy()

    for hand_landmarks in result.hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

    if result.gestures and result.gestures[0]:
        gesture = result.gestures[0][0]
        label = f"{gesture.category_name} ({gesture.score:.2f})"
        cv2.putText(
            annotated_frame, label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
        )

    cv2.putText(
        annotated_frame, f"FPS: {fps:.1f}", (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
    )
    cv2.imshow("Gesture recognition", annotated_frame)

camera_index = 4
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    raise RuntimeError(f"Could not open webcam {camera_index}")

previous_frame_time = None
try:
    with GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            success, frame = cap.read()
            if not success:
                print("Could not read a webcam frame")
                break

            frame_time = time.perf_counter()
            fps = 0.0
            if previous_frame_time is not None:
                elapsed_seconds = frame_time - previous_frame_time
                if elapsed_seconds > 0:
                    fps = 1.0 / elapsed_seconds
            previous_frame_time = frame_time

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=rgb_frame
            )
            timestamp_ms = time.monotonic_ns() // 1_000_000
            result = recognizer.recognize_for_video(mp_image, timestamp_ms)
            display_result(frame, result, fps)

            if cv2.waitKey(1) & 0xFF == 27:
                break
finally:
    cap.release()
    cv2.destroyAllWindows()