import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision.gesture_recognizer_result import (
    GestureRecognizerResult,
)

model_path = 'models/gesture_recognizer.task'

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles


def display_image_with_gesture_and_hand_landmarks(
    image, results: GestureRecognizerResult
):
    """Display the gesture category and hand landmarks on the image."""
    annotated_image = image.copy()

    if results.gestures and results.hand_landmarks:
        top_gesture = results.gestures[0][0]
        title = f"{top_gesture.category_name} ({top_gesture.score:.2f})"

        for hand_landmarks in results.hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

        cv2.putText(
            annotated_image,
            title,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Show", annotated_image)

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

camera_index = 0
cap = cv2.VideoCapture(camera_index)


with GestureRecognizer.create_from_options(options) as recognizer:
    while True:
        success, img = cap.read()

        if not success:
            print("Ignoring empty frame")
            continue

        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        timestamp_ms = time.monotonic_ns() // 1_000_000
        gesture_recognition_result = recognizer.recognize_for_video(
            mp_image, timestamp_ms
        )
        #print(gesture_recognition_result)
        display_image_with_gesture_and_hand_landmarks(img, gesture_recognition_result)

        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()
