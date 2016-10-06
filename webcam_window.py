import cv2


def ShowWebcam():
	try:
		cap = cv2.VideoCapture(0)

		# check if open
		if not cap.isOpened():
			raise IOError("Cannot open webcam")
			
		while True:
			ret, frame = cap.read()
			frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
			cv2.imshow('Input', frame)
			
			c = cv2.waitKey(1)
			if c == 27:
				break
	finally:
		cap.release()
		cv2.destroyAllWindows()

if __name__ == "__main__":
	ShowWebcam()