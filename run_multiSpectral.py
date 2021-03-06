import stapipy as st
import cv2
import numpy as np
import pandas as pd
import threading
import serial
import time
from time import gmtime, strftime
import os
import sys

from filterwheel import FW102C



DISPLAY_RESIZE_FACTOR = 0.3

class CMyCallback:
		"""
		Class that contains a callback function.
		"""

		def __init__(self):
				self._image = None
				self._lock = threading.Lock()

		@property
		def image(self):
				"""Property: return PyIStImage of the grabbed image."""
				duplicate = None
				self._lock.acquire()
				if self._image is not None:
						duplicate = self._image.copy()
				self._lock.release()
				return duplicate

		def datastream_callback(self, handle=None, context=None):
				"""
				Callback to handle events from DataStream.

				:param handle: handle that trigger the callback.
				:param context: user data passed on during callback registration.
				"""
				st_datastream = handle.module
				if st_datastream:
						with st_datastream.retrieve_buffer() as st_buffer:
								# Check if the acquired data contains image data.
								if st_buffer.info.is_image_present:
										# Create an image object.
										st_image = st_buffer.get_image()

										# Check the pixelformat of the input image.
										pixel_format = st_image.pixel_format
										pixel_format_info = st.get_pixel_format_info(pixel_format)

										# Only mono or bayer is processed.
										if not(pixel_format_info.is_mono or pixel_format_info.is_bayer):
												return

										# Get raw image data.
										data = st_image.get_image_data()

										# Perform pixel value scaling if each pixel component is
										# larger than 8bit. Example: 10bit Bayer/Mono, 12bit, etc.
										if pixel_format_info.each_component_total_bit_count > 8:
												nparr = np.frombuffer(data, np.uint16)
												division = pow(2, pixel_format_info
																			 .each_component_valid_bit_count - 8)
												nparr = (nparr / division).astype('uint8')
										else:
												nparr = np.frombuffer(data, np.uint8)

										# Process image for display.
										nparr = nparr.reshape(st_image.height, st_image.width, 1)

										# Perform color conversion for Bayer.
										if pixel_format_info.is_bayer:
												bayer_type = pixel_format_info.get_pixel_color_filter()
												if bayer_type == st.EStPixelColorFilter.BayerRG:
														nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_RG2RGB)
												elif bayer_type == st.EStPixelColorFilter.BayerGR:
														nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_GR2RGB)
												elif bayer_type == st.EStPixelColorFilter.BayerGB:
														nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_GB2RGB)
												elif bayer_type == st.EStPixelColorFilter.BayerBG:
														nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_BG2RGB)

										# Resize image and store to self._image.
										nparr = cv2.resize(nparr, None,
																			 fx=DISPLAY_RESIZE_FACTOR,
																			 fy=DISPLAY_RESIZE_FACTOR)
										self._lock.acquire()
										self._image = nparr
										self._lock.release()


st.initialize()
st_system = st.create_system()
st_device = st_system.create_first_device()
st_datastream = st_device.create_datastream()
st_datastream.start_acquisition()
st_device.acquisition_start()

my_callback = CMyCallback()
cb_func = my_callback.datastream_callback

# Register callback for datastream
callback = st_datastream.register_callback(cb_func)

cap = cv2.VideoCapture(1)

status = True
status_capturing = False
status_gerak = False

frameCount = 0
FolderName = ''
mean = None
first_frame = None

responses = {}

fwl = FW102C(port='COM6')
if not fwl.isOpen:
	print ("FWL INIT FAILED")
	sys.exit(2)
print ('**info',fwl.getinfo())
print ('**idn?',fwl.query('*idn?'))

while status:
	key_input = cv2.waitKey(1)

	if key_input == ord('q'):
		break

	output_image = my_callback.image
	
	if output_image is not None:
		cv2.imshow('image', output_image)

	#move detection area
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21,21), 0)
	if first_frame is None:
		time.sleep(10)
		first_frame = gray
		continue

	delta_frame = cv2.absdiff(first_frame, gray)

	if mean is None:
		mean = np.mean(delta_frame)
		continue
	print("delta", np.mean(delta_frame), mean)
	if np.mean(delta_frame) > mean+35 or np.mean(delta_frame) < mean-35:
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

	# cv2.imshow('frame', frame)
	# cv2.imshow('capturing', gray)
	cv2.imshow('delta', delta_frame)
	cv2.imshow('thresh', thresh_delta)

	#end move detection area

	if responses['moved'] == '1':
		status_capturing = True
		if status_gerak == False:
			FolderName = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
			os.chdir('data')
			os.mkdir(FolderName)
			os.chdir(FolderName)
		status_gerak = True

	if frameCount < 11 and status_capturing == True:
		if frameCount < 10 :
			print ('**pos={}'.format(frameCount + 1), fwl.command('pos={}'.format(frameCount + 1)))
			print ('**pos?', fwl.query('pos?'))

		if frameCount > 0:
				output_image = cv2.resize(output_image, (1024,1088))
				cv2.imwrite(str(frameCount) + '.png', output_image)
				with open('filter{}'.format(frameCount) + '.txt', 'w') as file:
						for data in output_image:
								np.savetxt(file, data)

		
				
		frameCount += 1

	if frameCount == 11:
		responses = {}
		# kirim sinyal ke arduino untuk jalankan conveyour
		# print("finish ?", fwl.close())
		frameCount = 0
		status_capturing = False
		status_gerak = False
		os.chdir('..')
		os.chdir('..')
		time.sleep(15)

st_device.acquisition_stop()
st_datastream.stop_acquisition()
