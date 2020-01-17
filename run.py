import stapipy as st
import cv2
import numpy as np
import threading
import matplotlib.pyplot as plt
import pickle
import serial
import time

from annv2c import NewPrediction

#communicate with arduino to read and write data
# port_arduino = 'COM 4'
# toArduino1 = serial.Serial(port_arduino,9600, timeout=1)
port_arduino2 = 'COM 5'
toArduino2 = serial.Serial(port_arduino2, 9600, timeout=1)

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
mean = None
first_frame = None
responses = {}

status = True
while status:
    output_image = my_callback.image
    print('output_image',output_image)
    if output_image is not None:
        cv2.imshow('image', output_image)
    key_input = cv2.waitKey(1)

    #motion detection. this happens, if camera detection motion of palm fruit
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
    print(np.mean(delta_frame))
    print(np.mean(delta_frame))
    if np.mean(delta_frame) > mean+10 or np.mean(delta_frame) < mean-10:
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
    cv2.imshow('capturing', gray)
    cv2.imshow('delta', delta_frame)
    cv2.imshow('thresh', thresh_delta)

    capturing = None

    print(responses)
    if responses['moved'] == '1':
        time.sleep(1)
        # conveyor_stoped = toArduino2.write(str.encode(responses['moved']))

        conveyor_stoped = toArduino2.write(str.encode('0'))
        print('conveyor stop moving')

        status = False
        rescale_frame = cv2.resize(output_image, (1024,1088))
        h,w,l = rescale_frame.shape
        result_array = np.zeros((h,231,1))
        start_time = datetime.now()

        # capture frame per second
        for _ in range(50):
            ret, frame = cap.read()
            rescale_frame = cv2.resize(frame, (1024, 1088))
            crop_frame = rescale_frame[:, 370:601]
            result_array = np.append(result_array, crop_frame, axis=2)

        result_array = result_array[:,:,1:101]

        # Modify matrix of white reference and dark reference
        file_wr = 'wr.mat'
        file_blk = 'blk.mat'
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

        h1,w1,l1 = result_array.shape

        for i in range(l1):
            temp = np.subtract(result_array[:,:,i],blk)
            m, n = temp.shape
            for s in range(m):
                for t in range(n):
                    if temp[s][t] < 0:
                        temp[s][t] = 0
            result_array[:,:,i] = np.divide(temp, y)

        result_arrayv2 = np.zeros((100,231,1088))
        Ax, Ay, r = result_array.shape
        for i in range(Ax):
            for z in range(r):
                result_arrayv2[z,:,i] = result_array[i,:,z]

        print('dimension of array is {}'.format(result_arrayv2.shape))
        mean = []

        for i in range(1088):
            n = 1087-i
            res = result_arrayv2[:,:,n][40:55, 100:125]
            mean.append(np.mean(res))

        plt.plot(mean)
        end_time = datetime.now()
        time_needed = end_time - start_time
        print('the time needed is {} seconds'.format(time_needed.seconds))

        filename = "parameterValue"
        #Prediction
        prediction = NewPrediction(filename, np.mean(mean))
        result = prediction.predict()
        print("Result Prediction is {}".format(result))

        # Sending result response to arduino to turn on conveyor
        # conveyor_moved = toArduino2.write(str.encode(mv_dtc['status_off']))
        print('conveyor moved {}'.format(conveyor_moved))
        time.sleep(3)

        #move arm
        time.sleep(1)
        # arm_moved = toArduino1.write( str.encode(result['index']))
        print('arm_moved'.format(arm_moved))
        
        print ("Program done")

        time.sleep(3)
        status = True
        plt.show()
    if key_input == ord('q'):
        break

st_device.acquisition_stop()
st_datastream.stop_acquisition()