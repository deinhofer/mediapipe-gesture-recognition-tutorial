---
marp: true
---

<!--
theme: fhtw
title: 'Assistive Technologies Platformen - Hand Gesture Recognition'
footer: 'Assistive Technologies Platformen - Hand Gesture Recognition'
paginate: true
_paginate: skip
_footer: ''
_class: lead
-->

# Assistive Technologies Platformen
## Hand Gesture Recognition

---

# Hand Gesture Recognition
## A step-by-step Python tutorial

From webcam frames to a labeled, landmark-annotated video window with OpenCV and MediaPipe Tasks.

**You will build:** the workflow in `gesture-recognition-sync-drawlandmarks.py`, then add a frame-rate overlay.

---

## What we are building

For every frame from the webcam, the program will:

1. Read a frame with OpenCV.
2. Convert the frame from OpenCV's BGR channel order to RGB.
3. Ask MediaPipe's Gesture Recognizer to process the frame.
4. Draw hand landmarks, their connections, the gesture name, and its score.
5. Show the result and an estimated frame rate.

The recognizer runs in **video mode**, so each frame also needs a timestamp that increases over time.

---

## Before you start

From the repository root, install the project dependencies and run the example:

```bash
pip install -r requirements.txt
python gesture-recognition-sync-drawlandmarks.py
```

The model bundle must be available at `models/gesture_recognizer.task`. The camera number in `cv2.VideoCapture(...)` depends on your computer; this project currently uses `2`. Try `0` if that is not your webcam.

The code below uses the MediaPipe Tasks API available in MediaPipe 1.1.1. In particular, its landmark drawing utilities live under `mp.tasks.vision`, not the older `mp.solutions` API.

---

## Step 1 — Import the tools and select the model

Add these imports and settings at the top of the program:

```python
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision.gesture_recognizer_result import (
    GestureRecognizerResult,
)

model_path = "models/gesture_recognizer.task"
```

`cv2` handles webcam capture, color conversion, text, and windows. `mediapipe` provides the recognizer. `time` supplies timestamps and the clock used later to estimate FPS. The model path points to the downloaded task bundle.

---

## Step 2 — Open the webcam

Create a capture object before starting the processing loop:

```python
camera_index = 2  # Change this to the webcam index for your computer.
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    raise RuntimeError(f"Could not open webcam {camera_index}")
```

`VideoCapture` opens a camera device. Index `0` is often the default camera, but the correct index varies. Checking `isOpened()` catches a bad index or unavailable camera before the loop begins.

---

## Step 3 — Read one frame at a time

Inside the loop, ask OpenCV for the next frame:

```python
success, frame = cap.read()

if not success:
    print("Could not read a webcam frame")
    break
```

`read()` returns two values:

- `success`: whether a frame was captured.
- `frame`: the image as a NumPy array when capture succeeds.

Stop on a failed read rather than sending an empty frame to MediaPipe. A continuously failing camera would otherwise make the loop spin without producing useful output.

---

## Step 4 — Know the image formats

An image is pixel data, but APIs may expect different channel orders or representations:

| Name | What it means here |
|---|---|
| **BGR** | Three-channel NumPy image in Blue, Green, Red order. This is what OpenCV camera capture returns by default. |
| **RGB** | Three-channel NumPy image in Red, Green, Blue order. This is the channel order we pass to MediaPipe. |
| **SRGB** | The MediaPipe image-format label for standard RGB image data. It describes how to interpret the pixel bytes; it does not reorder them. |
| **Grayscale / BGRA / RGBA** | Other common layouts: one intensity channel, or four channels with alpha. They are not the three-channel format used in this example. |

**Color space** and **channel order** are related but different ideas. Here, the practical issue is that the same three bytes are interpreted in different orders by OpenCV and MediaPipe.

---

## Step 5 — Convert BGR to RGB

Convert the captured frame before wrapping it in a MediaPipe `Image`:

```python
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_frame,
)
```

OpenCV's `frame` is BGR, while the MediaPipe Tasks image is declared as `SRGB` and therefore expects RGB-ordered data. The `SRGB` label does **not** perform conversion for us; `cv2.cvtColor` changes the channel order.

Without the conversion, red and blue are swapped when the image is interpreted. That can give the recognizer incorrectly colored input and makes the format declaration inaccurate.

---

## Step 6 — Configure the gesture recognizer

Select the MediaPipe Tasks APIs, then create options in video mode:

```python
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
RunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
)
```

`BaseOptions` tells MediaPipe which trained model to load. `VIDEO` mode is intended for a sequence of timestamped images. We use synchronous `recognize_for_video()` below: it returns the result for the current frame before the loop continues.

---

## Step 7 — Give each frame a timestamp

For every frame, obtain milliseconds from a monotonic clock:

```python
timestamp_ms = time.monotonic_ns() // 1_000_000
```

`monotonic_ns()` is not affected by changes to the computer's wall clock. Dividing nanoseconds by one million converts the value to milliseconds, as expected by the video API.

Video-mode timestamps must increase from frame to frame. Do not use a constant timestamp, and avoid wall-clock time such as `time.time()` for measuring intervals.

---

## Step 8 — Recognize a gesture

Create the recognizer once, outside the frame loop, then submit each frame:

```python
with GestureRecognizer.create_from_options(options) as recognizer:
    # Inside the webcam loop, after creating mp_image and timestamp_ms:
    result = recognizer.recognize_for_video(mp_image, timestamp_ms)
```

The `with` block manages the recognizer's native resources. Creating it once avoids reloading the model for every frame. `recognize_for_video` processes one image and returns a `GestureRecognizerResult`.

The result contains gesture classifications and hand landmarks. Each detected hand has its own entries in these lists.

---

## Step 9 — Get the gesture name and score

The first classification for the first detected hand can be read like this:

```python
if result.gestures and result.gestures[0]:
    top_gesture = result.gestures[0][0]
    label = f"{top_gesture.category_name} ({top_gesture.score:.2f})"
```

`category_name` is the predicted gesture label. `score` is the model's classification score; formatting it with `.2f` shows two decimal places.

The nested list is organized by detected hand, then by category. This snippet displays the highest-ranked category for the **first** hand; a multi-hand application can loop through the hands and their categories.

---

## Step 10 — Draw landmarks and connections

In MediaPipe 1.1.1, use the Tasks vision drawing tools and hand connection definitions:

```python
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

for hand_landmarks in result.hand_landmarks:
    mp_drawing.draw_landmarks(
        annotated_frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style(),
    )
```

Each detected hand has 21 normalized landmarks. `HAND_CONNECTIONS` describes which landmark pairs form the hand skeleton. The style helpers provide the default point and line styles.

The drawing helper accepts the landmarks directly; there is no need to convert them into a legacy protobuf `NormalizedLandmarkList`. It draws onto the supplied image, so use a copy if the original frame should remain unchanged.

---

## Step 11 — Prepare the annotated frame

Start with a copy, draw any landmarks, and add the gesture label when one is available:

```python
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
    top_gesture = result.gestures[0][0]
    label = f"{top_gesture.category_name} ({top_gesture.score:.2f})"
    cv2.putText(
        annotated_frame, label, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
    )
```

The text origin `(10, 30)` is near the upper-left corner. OpenCV text colors use BGR order too; `(0, 255, 0)` is green in either RGB or BGR because its red and blue components are zero.

---

## Step 12 — Calculate the frame rate

To estimate the displayed stream rate, measure the time between successful frame reads:

```python
previous_frame_time = None

# After a successful cap.read():
frame_time = time.perf_counter()
fps = 0.0
if previous_frame_time is not None:
    elapsed_seconds = frame_time - previous_frame_time
    if elapsed_seconds > 0:
        fps = 1.0 / elapsed_seconds
previous_frame_time = frame_time
```

`perf_counter()` is a high-resolution monotonic clock for measuring elapsed time. One frame divided by the seconds between frames gives frames per second: **FPS = 1 / elapsed_seconds**.

The first frame has no previous timestamp, so its FPS starts at `0.0`. This is an interval estimate, so it can fluctuate from frame to frame; smoothing multiple intervals can make the display steadier.

---

## Step 13 — Draw the FPS and show the window

Add an FPS overlay on every frame, even if no hand is detected:

```python
cv2.putText(
    annotated_frame,
    f"FPS: {fps:.1f}",
    (10, 65),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2,
)
cv2.imshow("Gesture recognition", annotated_frame)
```

This FPS estimates the rate at which successful camera frames reach the loop. It includes recognition and display work between reads, as well as camera timing; it is not a benchmark of the model alone. `imshow` displays the current annotated image.

---

## Step 14 — Let the user stop the program

Check for Escape after showing each frame, then always release the camera:

```python
if cv2.waitKey(1) & 0xFF == 27:
    break

# After the recognizer block:
cap.release()
cv2.destroyAllWindows()
```

`waitKey` gives the window time to update and reads a key press. The `& 0xFF` pattern makes the key comparison work across platforms; `27` is Escape. Releasing the camera and closing OpenCV windows are important cleanup steps.

---

## Complete program (1/3): setup

This first part combines the imports and recognizer configuration:

```python
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
```

---

## Complete program (2/3): draw the result

Add this function after the setup. It draws all detected hands, the first hand's top gesture, and the measured FPS:

```python
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
```

---

## Complete program (3/3): process webcam frames

Add the webcam loop below the function:

```python
camera_index = 2
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
```

## Run, observe, and experiment

Run the finished script and try these small experiments:

- Change the camera index if the webcam window is empty.
- Cover or move your hand and watch the landmarks follow it.
- Try the model's supported gestures and compare labels and scores.
- Temporarily remove `cv2.cvtColor` to see why correct channel ordering matters.
- Compare the FPS while the camera is idle and while recognition is running.

Keep the task model at the configured path, and press **Escape** to close the window cleanly.
