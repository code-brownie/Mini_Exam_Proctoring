from flask import Flask, render_template, Response
import cv2
import dlib
from mtcnn.mtcnn import MTCNN
import math

app = Flask(__name__)

def calculate_roll_angle(landmarks):
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

def detect_faces_and_gaze():
    face_detector = MTCNN()
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    roll_angle_threshold = 15

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set the desired frames per second

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

                roll_angle = calculate_roll_angle(landmarks)
                if left_eye_closed or right_eye_closed or abs(roll_angle) > roll_angle_threshold:
                    cv2.putText(frame, "Error: Look at the screen!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_faces_and_gaze(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
