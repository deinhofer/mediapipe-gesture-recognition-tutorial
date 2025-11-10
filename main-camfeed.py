import cv2
import mediapipe as mp
import time

# init camera object
cap=cv2.VideoCapture(0)

while True:
    # read frame from camera
    success,img=cap.read()

    if success:
        # show frame in window if reading was successful
        cv2.imshow("Camera Feed",img)
        

    # wait for pressing ESC to break the loop
    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
