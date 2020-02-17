import cv2
import numpy as np 
import matplotlib.pyplot as plt 



filename1 = "data/sampel 1/{}.png"
filename2 = "data/sampel 2/{}.png"
filename3 = "data/sampel 3/{}.png"
filename4 = "data/sampel 4/{}.png"
filename5 = "data/sampel 5/{}.png"

mean1 = []
mean2 = []
mean3 = []
mean4 = []
mean5 = []

wavelength = np.array([520, 680, 710, 740, 770, 800, 830, 860, 880, 920])

for i in range(1,11):
	data = cv2.imread(filename1.format(i))
	data = cv2.resize(data, (231, 1088))
	print(data.shape)
	mean1.append(np.mean(data))

for i in range(1,11):
	data = cv2.imread(filename2.format(i))
	data = cv2.resize(data, (231, 1088))
	mean2.append(np.mean(data))

for i in range(1,11):
	data = cv2.imread(filename3.format(i))
	data = cv2.resize(data, (231, 1088))
	mean3.append(np.mean(data))

for i in range(1,11):
	data = cv2.imread(filename4.format(i))
	data = cv2.resize(data, (231, 1088))
	mean4.append(np.mean(data))

for i in range(1,11):
	data = cv2.imread(filename5.format(i))
	data = cv2.resize(data, (231, 1088))
	mean5.append(np.mean(data))

plt.plot(wavelength, mean1)
plt.plot(wavelength, mean2)
plt.plot(wavelength, mean3)
plt.plot(wavelength, mean4)
plt.plot(wavelength, mean5)
plt.show()



# print(mean)