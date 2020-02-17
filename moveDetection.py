import cv2
import numpy as np 
import time

first_frame = None
mean = None
responses = {}

cap = cv2.VideoCapture(1)

while True:
	key_input = cv2.waitKey(1)

	if key_input == ord('q'):
		break
	
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21,21), 0)
	
	if first_frame is None:
		time.sleep(10)
		first_frame = gray
		continue

	delta_frame = cv2.absdiff(first_frame, gray)
	# print("delta_frame", delta_frame)
	if mean is None:
		mean = np.mean(delta_frame)
		continue
	print(mean, np.mean(delta_frame))
	if np.mean(delta_frame) > mean+20 or np.mean(delta_frame) < mean-20:
		responses['moved'] = '1'
		print('object moved')
	else:
		responses['moved'] = '0'
		print('no object moved')

	thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
	thresh_delta = cv2.dilate(thresh_delta, None, iterations=0)
	cnts, __ = cv2.findContours(thresh_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	for contour in cnts:
		if cv2.contourArea(contour)<10000:
			continue
		(x,y,w,h) = cv2.boundingRect(contour)
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)

	cv2.imshow('frame', frame)
	# cv2.imshow('capturing', gray)
	cv2.imshow('delta', delta_frame)
	cv2.imshow('thresh', thresh_delta)