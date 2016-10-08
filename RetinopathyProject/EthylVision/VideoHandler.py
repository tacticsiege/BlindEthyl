import cv2
from orb_feature_demo import detect_corners as orb
from face_detection_demo import detect_face

class VideoHandler(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.paused = False
        self.orb = False
        self.face = False
        self.frame = None
        self.scaling_factor = 0.5

        cv2.namedWindow('Video')

    def start(self):
        while True:
            is_running = not self.paused
            if is_running or self.frame is None:
                ret, frame = self.cap.read()

                frame = cv2.resize(frame, None,
                                   fx=self.scaling_factor,
                                   fy=self.scaling_factor,
                                   interpolation=cv2.INTER_AREA)
                if not ret:
                    break

                self.frame = frame.copy()

            img = self.frame.copy()
            if self.orb:
                img = orb(img)
            if self.face:
                img = detect_face(img)

            cv2.imshow('Video', img)
            ch = cv2.waitKey(1)
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('o'):
                self.orb = not self.orb
            if ch == ord('f'):
                self.face = not self.face
            if ch == ord('+'):
                self.scaling_factor += 0.1
            if ch == ord('-'):
                self.scaling_factor -= 0.1
            if ch == 27:
                break

if __name__ == "__main__":
    VideoHandler().start()
