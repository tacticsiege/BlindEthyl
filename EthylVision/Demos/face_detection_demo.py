import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt.xml')


def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in face_rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return image


def show_webcam():
    try:
        cap = cv2.VideoCapture(0)

        # check if open
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            frame = detect_face(frame)

            cv2.imshow('Input', frame)

            c = cv2.waitKey(1)
            if c == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    show_webcam()
