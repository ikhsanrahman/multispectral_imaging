import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import pickle
import serial
import time
from gtts import gTTS
import os
from datetime import datetime
from tkinter import *

dig = load_digits()
onehot_target = pd.get_dummies(dig.target)
x_train, x_val, y_train, y_val = train_test_split(dig.data, onehot_target, test_size=0.1, random_state=20)

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def sigmoid_derv(s):
    return s * (1 - s)

def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

def cross_entropy_derv(pred, real):
    n_samples = real.shape[0]
    res = pred - real

    return res/n_samples

def cross_entropy(pred, real): #here, cross entropy is error function, it's similiar to MSE
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp)/n_samples
    return loss

def getPrediction(inputData, w1, b1, w2, b2, w3, b3):
    z1 = np.dot(inputData, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)
    z3 = np.dot(a2, w3) + b3
    a3 = softmax(z3)
    return a3

def Window(data):
    window = Tk() 
    window.title("classification of Palm Fruit") 
    window.geometry("1000x300") 
    lbl = Label(window, text=data, bg="red", fg="white") 
    lbl.config(font=("Courier", 70)) 
    lbl.pack(fill=X, padx=10, pady=100) 
    window.mainloop() 


class MyNN:
    def __init__(self, x, y):

        self.x = x
        neurons = 256
        self.lr = 0.5
        ip_dim = x.shape[1]
        op_dim = y.shape[1]

        self.w1 = np.random.randn(ip_dim, neurons)
        self.b1 = np.zeros((1, neurons))
        self.w2 = np.random.randn(neurons, neurons)
        self.b2 = np.zeros((1, neurons))
        self.w3 = np.random.randn(neurons, op_dim)
        self.b3 = np.zeros((1, op_dim))
        # self.w1 = np.zeros((ip_dim, neurons))
        # self.b1 = np.zeros((1, neurons))
        # self.w2 = np.zeros((neurons, neurons))
        # self.b2 = np.zeros((1, neurons))
        # self.w3 = np.zeros((neurons, op_dim))
        # self.b3 = np.zeros((1, op_dim))
        self.y = y

    def feedforward(self):
        z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = sigmoid(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(z2)
        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = softmax(z3)
        return self.a3
        
    def backprop(self):
        loss = cross_entropy(self.a3, self.y)
        print('Error :', loss)
        a3_delta = cross_entropy_derv(self.a3, self.y) # w3
        z2_delta = np.dot(a3_delta, self.w3.T)
        a2_delta = z2_delta * sigmoid_derv(self.a2) # w2
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * sigmoid_derv(self.a1) # w1

        self.w3 -= self.lr * np.dot(self.a2.T, a3_delta)
        self.b3 -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
        self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
        self.b2 -= self.lr * np.sum(a2_delta, axis=0)
        self.w1 -= self.lr * np.dot(self.x.T, a1_delta)
        self.b1 -= self.lr * np.sum(a1_delta, axis=0)
        # dbfile = open('konstanta', 'ab')

    def getConstant(self):
        result = {
            'w3': self.w3,
            'b3': self.b3,
            'w2': self.w2,
            'b2': self.b2,
            'w1': self.w1,
            'b1': self.b1,
        }
        return result


    def predict(self, data):
        self.x = data
        return self.feedforward()
        return self.a3.argmax()
			
# model = MyNN(x_train/16.0, np.array(y_train))

# epochs = 1
# for x in range(epochs):
#     model.feedforward()
#     model.backprop()
		
# def get_acc(x, y):
#     acc = 0
#     for xx,yy in zip(x, y):
#         s = model.predict(xx)
#         if s == np.argmax(yy):
#             acc +=1
#     return acc/len(x)*100

# print("Training accuracy : ", get_acc(x_train/16, np.array(y_train)))
# print("Test accuracy : ", get_acc(x_val/16, np.array(y_val)))


class DataProcessing:

    def __init__(self):
        self.out_data = ''

    def newData(self, path):
        data = pd.read_excel(path)
        # data_testingv1 = data.drop(data.columns[0], axis=1)
        # new_datav2 = data_testingv1.drop(data_testingv1.columns[0], axis=1) #drop row index 0
        # inp_data = np.array(new_datav2)
        # print(data, 'afafafafa')
        inp_data = np.array(data)

        return inp_data

    def processData(self, path):
        self.data = pd.read_excel(path)
        self.new_datav1 = self.data.drop(self.data.columns[0], axis=1) #drop column index 0

        self.new_datav2 = self.new_datav1.drop(self.new_datav1.columns[0], axis=1) #drop row index 0

        self.inp_data = np.array(self.new_datav2)

        # self.out_data = np.random.randn(self.inp_data.shape[0], 4)
        # print(self.out_data, 'out')

        labels = np.array([0]*8 + [1]*10 + [2]*5 + [3]*3)
        
        one_hot_labels = np.zeros((self.inp_data.shape[0], 4))

        for i in range(self.inp_data.shape[0]):
            one_hot_labels[i, labels[i]] = 1

        result = {}

        result['input_data'] = self.inp_data
        result['actual_output'] =  one_hot_labels

        x_train = self.inp_data
        y_train = one_hot_labels
        x_val = self.inp_data[10:12]
        y_val = one_hot_labels[10:12]

        # self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.inp_data, one_hot_labels, test_size=0.1, random_state=20)

        result['x_train'] = x_train
        result['x_val'] = x_val
        result['y_train'] = y_train
        # result['y_val'] = y_val

        return result

# modelProcess = DataProcessing()
# # path_process = '/home/ikhsan/testing/ripeness.xlsx'
# # path_process = '/home/ikhsan/Projects/palm-fruit-ann/data_sawit1.xlsx'
# path_process = 'data_sawit1.xlsx'
# path_testing = 'matang1.xlsx'
# # path_testing = '/home/ikhsan/testing/data_sawit_testing.xlsx'
# # path_testing = '/home/ikhsan/Projects/palm-fruit-ann/matang1.xlsx'
# # path_testing = 'mentah_2.xlsx'

# process = modelProcess.processData(path_process)
# testing = modelProcess.newData(path_testing)


# x_train = process['x_train']
# # print(x_train, 'train')
# y_train = process['y_train']
# x_val = process['x_val']
# # x_val = np.random.randn(2,1088)
# # x_val = testing
# # y_val = np.array([0, 1, 0, 0])


# model = MyNN(x_train, np.array(y_train))

# time_start = datetime.now()
# epochs = 4000
# for x in range(epochs):
#     model.feedforward()
#     model.backprop()
# time_end = datetime.now()
# time_needed = time_end - time_start
# print('waktu yang dibutuhkan ', time_needed.seconds )

# getConstant = model.getConstant()
# filename = "parameterValue"
# outfile = open(filename, 'wb')
# pickle.dump(getConstant, outfile)
# outfile.close()


# result = model.predict(x_val)


# ------------------------------------- New Data ------------------------------------

class NewPrediction:
    def __init__(self, filename, newInput):
        self.filename = filename
        self.newInput = newInput

    def getPrediction(self, inputData, w1, b1, w2, b2, w3, b3):
        z1 = np.dot(inputData, w1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, w2) + b2
        a2 = sigmoid(z2)
        z3 = np.dot(a2, w3) + b3
        a3 = softmax(z3)
        return a3

    def predict(self):
        infile = open(self.filename, 'rb')
        param = pickle.load(infile)
        result = self.getPrediction(self.newInput, param['w1'], param['b1'], param['w2'], param['b2'], param['w3'], param['b3'])
        print(result[0], result.shape, "result")
        (m,i) = max((v, i) for i,v in enumerate(result[0]))

        outcome = {}
        if i in range(2):
            outcome['message'] = 'Mentah'
            outcome['index'] = '0'

        if i in range(2,4):
            outcome['message'] = 'Matang'
            outcome['index'] = '1'
        
        # Window(outcome['message'])

        return outcome  




# filename = "parameterValue"
# newPredict = NewPrediction(filename, x_val)
# result = newPredict.predict()
# print('resutl',result)
# infile = open(filename, 'rb')
# newData = pickle.load(infile)
# result = getPrediction(x_val, newData['w1'], newData['b1'], newData['w2'], newData['b2'], newData['w3'], newData['b3'],)

# (m,i) = max((v, i) for i,v in enumerate(result[0]))

# outcome = {}
# if i in range(2):
#     outcome['message'] = 'Mentah'
#     outcome['index'] = '0'
    


# if i in range(2,4):
#     outcome['message'] = 'Matang'
#     outcome['index'] = '1'

# if outcome['message'] == 'Mentah':
#     os.system("mpg321 mentah.mp3")

# if outcome['message'] == 'Matang':
#     os.system("mpg321 matang.mp3")

# Window(outcome['message'])
# # print('prediction result', result)
# print('actual result', y_val   )
# # print(outcome)


# data = serial.Serial('/dev/ttyACM0',9600, timeout=1) #nama port, baurate   #ketika run arduino harus konek dan cocokin kedua item ini


# # # for i in range(10)




# time.sleep(2)
# status = data.write( str.encode(outcome['index'])) #data yg terinput akan ditampilkan LCD dan diakhiri enter
# # print(status, i)

# # if not result['index'] == '1' and not result['index'] == '2':
# #   run = False
# print ("Program done")