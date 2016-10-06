import cv2
import numpy as np
import argparse

def detect_corners(filename):
	img = cv2.imread(filename)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	orb = cv2.ORB()
	
	keypoints = orb.detect(gray, None)
	keypoints, descriptors = orb.compute(gray, keypoints)
	
	img = cv2.drawKeypoints(img, keypoints, color=(0,255,0), flags=0)
	
	return img
	
if __name__ == "__main__":	
	parser = argparse.ArgumentParser(description='Process ORB features.')
	parser.add_argument('filename', metavar='f', help='Filename of image to process.')
	args = parser.parse_args()
	#print args.filename
	img = detect_corners(args.filename)
	img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
	cv2.imshow('ORB Features', img)
	cv2.waitKey()
	
