import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

import time

model_path = 'models/gesture_recognizer.task'

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print('gesture recognition result: {}'.format(result))
    # show frame in window if reading was successful
    #cv2.imshow("Camera Feed", out

def display_images_with_gesture_and_hand_landmarks(image, results):
    """Displays a batch of images with the gesture category and its score along with the hand landmarks."""
    # Images and labels.

    if not results.gestures or not results.hand_landmarks:
        return
    
    top_gesture = results.gestures[0][0]  # Get the top gesture from the first hand.
    multi_hand_landmarks_list = results.hand_landmarks

    # Size and spacing.
    FIGSIZE = 13.0
    SPACING = 0.1

    # Display gestures and hand landmarks.

    title = f"{top_gesture.category_name} ({top_gesture.score:.2f})"
    annotated_image = image.copy()

    for hand_landmarks in multi_hand_landmarks_list:
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])

        mp_drawing.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())

    cv2.putText(annotated_image, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Show", annotated_image)

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)


# init camera object
import cv2
cap=cv2.VideoCapture(0)


with GestureRecognizer.create_from_options(options) as recognizer:
  # The detector is initialized. Use it here.
  # ...
    while True:
        # read frame from camera
        success, img = cap.read()

        if not success:
            print("Ignoring empty frame")
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        gesture_recognition_result = recognizer.recognize_for_video(mp_image, int(time.time_ns() / 1000000))
        print(gesture_recognition_result)
        display_images_with_gesture_and_hand_landmarks(img, gesture_recognition_result)
        #cv2.imshow("Camera Feed", img)

        # wait for pressing ESC to break the loop
        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyAllWindows()
            break

    




