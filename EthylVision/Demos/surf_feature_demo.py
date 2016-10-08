import cv2
import numpy as np
import argparse

def detect_corners(filename):
	img = cv2.imread(filename)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	surf = cv2.SURF()
	
	surf.hessianThreshold = 15000
	
	kp, des = surf.detectAndCompute(gray, None)
	
	img = cv2.drawKeypoints(img, kp, None, (0,255,0), 4)
	
	return img
	
if __name__ == "__main__":	
	parser = argparse.ArgumentParser(description='Process SURF detector.')
	parser.add_argument('filename', metavar='f', help='Filename of image to process.')
	args = parser.parse_args()
	#print args.filename
	img = detect_corners(args.filename)
	img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
	cv2.imshow('SURF detector', img)
	cv2.waitKey()
	
