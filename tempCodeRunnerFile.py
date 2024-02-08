import cv2
from mtcnn.mtcnn import MTCNN

def detect_faces_webcam():
    # Create an instance of MTCNN
    detector = MTCNN()

    # Open a connection to the webcam (usually, 0 represents the default camera)
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Detect faces in the frame
        faces = detector.detect_faces(frame)

        # Draw bounding boxes around the detected faces
        for face in faces:
            x, y, width, height = face['box']
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Display the result
        cv2.imshow('Face Detection - Webcam', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Call the detect_faces_webcam function
    detect_faces_webcam()
