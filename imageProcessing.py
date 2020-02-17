import cv2
import numpy as np
import time
import math
from datetime import datetime
import scipy.io as sc
import matplotlib.pyplot as plt

start_time = datetime.now()

def dataProcessing(data):
	result_array = np.zeros((1088,231,1))
	h,w,l = data.shape
	for i in range(l):
		crop_frame = data[:,:,i][0:1088,370:601].reshape(1088,231,1)
		result_array = np.append(result_array, crop_frame, axis=2)    		

	result_array = result_array[:,:,1:101]
	# Modify matrix of white reference and dark reference
	wr = sc.loadmat('wr.mat')['wr'].astype(int)
	blk = sc.loadmat('blk.mat')['blk'].astype(int)
	y = np.subtract(wr, blk)

	m, n = y.shape
	for s in range(m):
		for t in range(n):
			if y[s][t] < 0:
				y[s][t] = 0
			if y[s][t] == 0:
				y[s][t] = 1

	# Subtract the image with black reference
	h1,w1,l1 = result_array.shape

	for i in range(l1):
		temp = np.subtract(result_array[:,:,i],blk)
		m, n = temp.shape
		for s in range(m):
			for t in range(n):
				if temp[s][t] < 0:
					temp[s][t] = 0
				# if temp[s][t] > 5:
				# 	temp[s][t] = 5
		result_array[:,:,i] = np.divide(temp, y)

	wavelength = 800
	init = np.zeros((100,231,1088))
	Ax, Ay, r = result_array.shape
	for i in range(Ax):
		for z in range(r):
			init[z,:,i] = result_array[i,:,z]

	print('dimension of array is {}'.format(result_array.shape))
	# Crop Image by taking in the middle of image
	mean = []
	for i in range(1088):
		n = 1087-i
		# res = result_array[:,:,i][math.ceil(h2*0.25):math.ceil(h2*0.75), math.ceil(w2*0.25):math.ceil(w2*0.75) ]
		res = init[:,:,n][40:55, 100:125]
		mean.append(np.mean(res))
	return mean

data1 = sc.loadmat('MTH1_DEPAN.mat')['MTH1_DEPAN'].reshape(1088,1024,100)
data1 = data1.astype(int)

data2 = sc.loadmat('MTG1_DEPAN.mat')['MTG1_DEPAN'].reshape(1088,1024,100)
data2 = data2.astype(int)

output1 = dataProcessing(data1)
output2 = dataProcessing(data2)

plt.ylabel("Intensity of Reflectance")
plt.xlabel("wavelength")
plt.plot(output1, color="olive", label="ripe")
plt.plot(output2, color="skyblue", linestyle="dashed", label="unripe")

plt.show()