import cv2
import dlib
from mtcnn.mtcnn import MTCNN
import math

def calculate_roll_angle(landmarks):
    # Convert landmarks to list of (x, y) tuples
    landmarks_list = [(pt.x, pt.y) for pt in landmarks.parts()]
    
    left_eye = landmarks_list[36]
    right_eye = landmarks_list[45]

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    return math.degrees(math.atan2(dy, dx))

def are_eyes_closed(eye_landmarks):
    vertical_distance = eye_landmarks[5].y - eye_landmarks[1].y
    threshold = 5
    return vertical_distance < threshold

def detect_faces_and_gaze_webcam():
    face_detector = MTCNN()
    predictor_path = "C:\\Users\\Aman Kumar\\OneDrive\\Desktop\\cv2\\shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    face_detector_dlib = dlib.get_frontal_face_detector()
    roll_angle_threshold = 15  # Adjust this threshold as needed

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        faces = face_detector.detect_faces(frame)

        if len(faces) > 1:
            cv2.putText(frame, "Error: Two people not allowed!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            for face in faces:
                x, y, width, height = face['box']
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rect = dlib.rectangle(x, y, x + width, y + height)
                landmarks = predictor(gray_frame, rect)

                left_eye_landmarks = landmarks.parts()[36:42]
                right_eye_landmarks = landmarks.parts()[42:48]

                for i in range(0, 5):
                    cv2.line(frame, (left_eye_landmarks[i].x, left_eye_landmarks[i].y),
                             (left_eye_landmarks[i + 1].x, left_eye_landmarks[i + 1].y), (0, 255, 0), 2)
                    cv2.line(frame, (right_eye_landmarks[i].x, right_eye_landmarks[i].y),
                             (right_eye_landmarks[i + 1].x, right_eye_landmarks[i + 1].y), (0, 255, 0), 2)

                left_eye_closed = are_eyes_closed(left_eye_landmarks)
                right_eye_closed = are_eyes_closed(right_eye_landmarks)

                # Requirement 1: Check if the user is looking away using head pose (roll angle)
                roll_angle = calculate_roll_angle(landmarks)
                if left_eye_closed or right_eye_closed or abs(roll_angle) > roll_angle_threshold:
                    cv2.putText(frame, "Error: Look at the screen!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Face and Gaze Detection - Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_faces_and_gaze_webcam()
