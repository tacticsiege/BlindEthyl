import cv2
import numpy as np
import argparse

def detect_corners(filename):
	img = cv2.imread(filename)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	gray = np.float32(gray)
	
	dst = cv2.cornerHarris(gray, 4, 5, 0.04) # sharp corners
	#dst = cv2.cornerHarris(gray, 14, 5, 0.04) # soft corners
	
	# result is dilated for marking corners
	dst = cv2.dilate(dst, None)
	
	# use threshold
	img[dst > 0.01*dst.max()] = [0,0,0]
	
	return img
	
if __name__ == "__main__":	
	parser = argparse.ArgumentParser(description='Process Harris Corners.')
	parser.add_argument('filename', metavar='f', help='Filename of image to process.')
	args = parser.parse_args()
	#print args.filename
	img = detect_corners(args.filename)
	img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
	cv2.imshow('Harris Corners', img)
	cv2.waitKey()
	
