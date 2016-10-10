import cv2


class VideoHandler(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.paused = False
        self.frame = None

    def start(self):
        cv2.namedWindow('VideoHandler')
        while True:
            is_running = not self.paused
            if is_running or self.frame is None:
                ret, frame = self.cap.read()

                if not ret:
                    break

                self.frame = frame.copy()
            img = self.frame.copy()
            cv2.imshow('VideoHandler', img)

            ch = cv2.waitKey(1)
            if ch == 27:
                break

    def get_frame(self):
        ret, img = self.cap.read()

        if not ret:
            return None

        return img.copy()

if __name__ == "__main__":
    VideoHandler().start()
