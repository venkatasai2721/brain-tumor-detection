
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from sklearn import metrics
from tkinter import ttk

main = tkinter.Tk()
main.title("Convolutional Neural Network based Brain Tumor Detection") #designing main screen
main.geometry("1300x1200")

global filename
global accuracy
X = []
Y = []
global classifier
disease = ['Normal','Benign']

with open('Model/segmented_model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    segmented_model = model_from_json(loaded_model_json)
json_file.close()    
segmented_model.load_weights("Model/segmented_weights.h5")
segmented_model._make_predict_function()

def cropTumorRegion():
    img = cv2.imread('myimg.png')
    orig = cv2.imread('test1.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    min_area = 0.95*180*35
    max_area = 1.05*180*35
    result = orig.copy()
    life = 0
    for c in contours:
        area = cv2.contourArea(c)
        if life == 0:
            life = len(c)
        cv2.drawContours(result, [c], -1, (0, 0, 255), 10)
        if area > min_area and area < max_area:
            cv2.drawContours(result, [c], -1, (0, 255, 255), 10)
    return result, life    

def getTumorRegion(filename):
    global segmented_model
    img = cv2.imread(filename,0)
    img = cv2.resize(img,(64,64), interpolation = cv2.INTER_CUBIC)
    img = img.reshape(1,64,64,1)
    img = (img-127.0)/127.0
    preds = segmented_model.predict(img)
    preds = preds[0]
    print(preds.shape)
    orig = cv2.imread(filename,0)
    orig = cv2.resize(orig,(300,300),interpolation = cv2.INTER_CUBIC)
    cv2.imwrite("test1.png",orig)    
    segmented_image = cv2.resize(preds,(300,300),interpolation = cv2.INTER_CUBIC)
    cv2.imwrite("myimg.png",segmented_image*255)
    edge_detection, lifespan = cropTumorRegion()
    return segmented_image*255, edge_detection, lifespan
    

def uploadDataset(): #function to upload dataset
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def datasetPreprocessing():
    text.delete('1.0', END)
    global X
    global Y
    X.clear()
    Y.clear()
    if os.path.exists('Model/myimg_data.txt.npy'):
        X = np.load('Model/myimg_data.txt.npy')
        Y = np.load('Model/myimg_label.txt.npy')
    else:
        for root, dirs, directory in os.walk(filename+"/no"):
            for i in range(len(directory)):
                name = directory[i]
                img = cv2.imread(filename+"/no/"+name) #reading images
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #processing and normalization images
                img = cv2.resize(img, (64,64)) #resizing images
                im2arr = np.array(img) #extract features from images
                im2arr = im2arr.reshape(64,64,1)
                X.append(im2arr)
                Y.append(0)
                print(filename+"/no/"+name)

        for root, dirs, directory in os.walk(filename+"/yes"):
            for i in range(len(directory)):
                name = directory[i]
                img = cv2.imread(filename+"/yes/"+name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                img = cv2.resize(img, (64,64))
                im2arr = np.array(img)
                im2arr = im2arr.reshape(64,64,1)
                X.append(im2arr)
                Y.append(1)
                print(filename+"/yes/"+name)
                
        X = np.asarray(X)
        Y = np.asarray(Y)            
        np.save("Model/myimg_data.txt",X)
        np.save("Model/myimg_label.txt",Y)
    print(X.shape)
    print(Y.shape)
    print(Y)
    text.insert(END,"Total number of images found in dataset : "+str(len(X))+"\n")
    text.insert(END,"Total number of classes : "+str(len(set(Y)))+"\n\n")
    text.insert(END,"Class labels found in dataset : "+str(disease))       
    text.update_idletasks()
    cv2.imshow('Sample Processed Image ',cv2.resize(X[20],(200,200)))
    cv2.waitKey(0)
 
def trainTumorDetectionModel():
    global accuracy
    global classifier
    text.delete('1.0', END)
    YY = to_categorical(Y)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    x_train = X[indices]
    y_train = YY[indices]

    if os.path.exists('Model/model.json'):
        with open('Model/model.json', "r") as json_file:
           loaded_model_json = json_file.read()
           classifier = model_from_json(loaded_model_json)

        classifier.load_weights("Model/model_weights.h5")
        classifier._make_predict_function()           
    else:
        X_trains, X_tests, y_trains, y_tests = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)
        classifier = Sequential() 
        classifier.add(Convolution2D(32, 3, 3, input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3]), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(output_dim = 128, activation = 'relu'))
        classifier.add(Dense(output_dim = 2, activation = 'softmax'))
        print(classifier.summary())
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = classifier.fit(x_train, y_train, batch_size=16, epochs=10,validation_data=(X_tests, y_tests), shuffle=True, verbose=2)
        classifier.save_weights('Model/model_weights.h5')            
        model_json = classifier.to_json()
        with open("Model/model.json", "w") as json_file:
            json_file.write(model_json)
        f = open('Model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    f = open('Model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    accuracy = acc[9] * 100
    text.insert(END,'\n\nCNN Brain Tumor Model Generated. See black console to view layers of CNN\n\n')
    text.insert(END,"CNN Brain Tumor Prediction Accuracy on Test Images : "+str(accuracy)+"\n")
       


def tumorClassification():
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,1)
    XX = np.asarray(im2arr)
        
    predicts = classifier.predict(XX)
    print(predicts)
    cls = np.argmax(predicts)
    print(cls)
    if cls == 0:
        img = cv2.imread(filename)
        img = cv2.resize(img, (800,500))
        cv2.putText(img, 'Classification Result : '+disease[cls], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
        cv2.imshow('Classification Result : '+disease[cls], img)
        cv2.waitKey(0)
    if cls == 1:
        segmented_image, edge_image, lifespan = getTumorRegion(filename)
        img = cv2.imread(filename)
        img = cv2.resize(img, (800,500))
        cv2.putText(img, 'Classification Result : '+disease[cls], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
        #cv2.putText(img, 'Predicted Lifespan : '+str(lifespan)+" Months", (10, 75),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
        cv2.imshow('Classification Result : '+disease[cls], img)
        cv2.imshow("Tumor Extracted Image",segmented_image)
        cv2.imshow("Crop Image",edge_image)
        cv2.waitKey(0)
        
        
        

def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()

    accuracy = data['accuracy']
    loss = data['loss']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Training Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color = 'red')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.title('Bone Tumor CNN Model Training Accuracy & Loss Graph')
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Convolutional Neural Network based Brain Tumor Detection')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Brain Tumor Images Dataset", command=uploadDataset)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

preprocessButton = Button(main, text="Dataset Preprocessing & Features Extraction", command=datasetPreprocessing)
preprocessButton.place(x=430,y=550)
preprocessButton.config(font=font1) 

cnnButton = Button(main, text="Train CNN Brain Tumor Detection Model", command=trainTumorDetectionModel)
cnnButton.place(x=810,y=550)
cnnButton.config(font=font1) 

classifyButton = Button(main, text="Brain Tumor Prediction", command=tumorClassification)
classifyButton.place(x=50,y=600)
classifyButton.config(font=font1)

graphButton = Button(main, text="Training Accuracy Graph", command=graph)
graphButton.place(x=430,y=600)
graphButton.config(font=font1)

main.config(bg='turquoise')
main.mainloop()
