import cv2
import numpy as np
import argparse

def detect_corners(filename):
	img = cv2.imread(filename)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	sift = cv2.SIFT()
	#keypoints = sift.detect(gray, None)
	keypoints, descriptors = sift.detectAndCompute(gray, None)
	img = cv2.drawKeypoints(img, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	
	return img
	
if __name__ == "__main__":	
	parser = argparse.ArgumentParser(description='Process SIFT Features.')
	parser.add_argument('filename', metavar='f', help='Filename of image to process.')
	args = parser.parse_args()
	#print args.filename
	img = detect_corners(args.filename)
	img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
	cv2.imshow('SIFT features', img)
	cv2.waitKey()
	
