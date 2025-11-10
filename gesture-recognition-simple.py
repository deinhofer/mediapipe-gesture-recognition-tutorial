import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import time

model_path = 'models/gesture_recognizer.task'

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def display_image_with_gesture(image, results):
    """Displays the image with the gesture category."""
    if not results.gestures or not results.hand_landmarks:
        return
    
    #print(gesture_recognition_result)
    top_gesture = results.gestures[0][0]  # Get the top gesture from the first hand.
    multi_hand_landmarks_list = results.hand_landmarks # Get the landmarks list of the first hand.

    # Display gestures and hand landmarks.
    title = f"{top_gesture.category_name} ({top_gesture.score:.2f})"
    annotated_image = image.copy()

    # Write the gesture text on the image.
    cv2.putText(annotated_image, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Show", annotated_image)

# Create a gesture recognizer instance with the video mode:
# LIVE STREAM mode is for processing live video streams, such as from a webcam.
# VIDEO mode is for processing pre-recorded videos.
# In this example, we use VIDEO mode to process frames from the webcam in a loop for simplicity.
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)

# init camera object
import cv2
cap=cv2.VideoCapture(0)

with GestureRecognizer.create_from_options(options) as recognizer:
  # The detector is initialized. Use it here.
    while True:
        # read frame from camera
        success, img = cap.read()

        # check if frame reading was successful, otherwise break the loop
        if not success:
            print("Ignoring empty frame")
            break

        # convert the frame to mediapipe image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

        # perform gesture recognition on the frame
        gesture_recognition_result = recognizer.recognize_for_video(mp_image, int(time.time_ns() / 1000000))
        display_image_with_gesture(img, gesture_recognition_result)

        # wait for pressing ESC to break the loop
        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyAllWindows()
            break