import cv2
import mediapipe as mp
import numpy as np

# Helper: Calculate angle between 3 points
def calcAngle(hand, elbow, shoulder):
    #inputs are (x, y) points
    a = np.array(hand)
    b = np.array(elbow)
    c = np.array(shoulder)

    ba = a - b
    bc = c - b

    cosAngle = np.dot(ba, bc) / np.linalg.norm(ba) * np.linalg.norm(bc)
    angle = np.degrees(np.arccos(cosAngle))
   
    return angle

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Use right-side body landmarks
        shoulder = (lm[12].x, lm[12].y)
        elbow = (lm[14].x, lm[14].y)
        wrist = (lm[16].x, lm[16].y)
        hip = (lm[24].x, lm[24].y)
        ankle = (lm[28].x, lm[28].y)

        elbow_angle = calcAngle(shoulder, elbow, wrist)
        back_angle = calcAngle(shoulder, hip, ankle)

        # Logic for feedback
        if elbow_angle < 120:
            if 80 <= elbow_angle <= 100:
                if 160 <= back_angle <= 200:
                    feedback = "Good pushup!"
                else:
                    feedback = "Bad pushup – Keep your back straight."
            else:
                feedback = "Bad pushup – Elbow angle off."
        else:
            feedback = "Bad pushup – Bend elbows more."

        print(f"Elbow angle: {elbow_angle:.1f}° | Back angle: {back_angle:.1f}°")
        print(feedback)
        print("-" * 50)

    cv2.imshow('Pushup Form Check – Press Q to quit', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
