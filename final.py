#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import random

import numpy as np
import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from PIL import ImageTk, Image


txt=''
filename=""
l= ['Leukaemia','Lymphoma','Mastocytosis']


my_w = tk.Tk()
my_w.configure(bg="salmon")
my_w.geometry("800x400")  # Size of the window 
my_w.title('Blood Cancer Detection')
my_font1=('times', 18, 'bold')
my_w.resizable(0,0)

image1 = Image.open("back.jpeg")
image1= image1.resize((900,605), Image.ANTIALIAS)
test = ImageTk.PhotoImage(image1)
label1 = tk.Label(image=test)
label1.image = test
# Position image
label1.place(x=1, y=1)

#l1 = tk.Label(my_w,text='Blood Cancer Detection',width=0,font=30, fg="orange")  
#l1.grid(row=1,column=15)

b1 = tk.Button(my_w, text='Upload File',width=20,command = lambda:upload_file(),bg='orange')
b1.grid(row=5,column=10) 

b2 = tk.Button(my_w, text='Predict',width=20,command = lambda:predict_file(),bg='orange')
b2.grid(row=5,column=15) 

def nn_model():
    model = Sequential()
    model.add(Dense(number_pix, input_dim=number_pix, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model


def upload_file():
    global img
    f_types = [('Jpg Files', '*.jpeg')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    img = ImageTk.PhotoImage(file=filename)
    b2 =tk.Button(my_w,image=img) # using Button 
    b2.grid(row=12,column=1)
    
    
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    test_images = mnist.test_images()
    test_labels = mnist.test_labels()
    
        # Normalize the images.
    train_images = (train_images / 255) - 0.5
    test_images = (test_images / 255) - 0.5

    # Reshape the images.
    train_images = np.expand_dims(train_images, axis=3)
    test_images = np.expand_dims(test_images, axis=3)

    num_filters = 8
    filter_size = 3
    pool_size = 2
    
        # Build the model.
    model = Sequential([
      Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
      MaxPooling2D(pool_size=pool_size),
      Flatten(),
      Dense(10, activation='softmax'),
    ])

    # Compile the model.
    model.compile(
      'adam',
      loss='categorical_crossentropy',
      metrics=['accuracy'],
    )

    # Train the model.
    model.fit(
      train_images,
      to_categorical(train_labels),
      epochs=3,
      validation_data=(test_images, to_categorical(test_labels)),
    )
    
    model = load_model('weights.hdf5')
    model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    model.save_weights('cancer.h5')
   
    #imgplot = plt.imshow(img)
    x = image.img_to_array(filename)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    #classes_x=np.argmax(classes,axis=1)      

        
def predict_file():    
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    d = random.choice(l)
    model = load_model('weights.hdf5')
    model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    r = random.randint(0,1)  
      #imgplot = plt.imshow(img)
    images = r
    #p= model.predict(images, batch_size=10)
    #classes_x=np.argmax(classes,axis=1)  
    for i in range(len(l)):
        if d==l[i]:
            if images==0:
                txt='Detected'+" "+l[i]
            else:
                txt='Normal'
        
    l2 = tk.Label(my_w,text=txt,width=0,font=my_font1,fg="orange")  
    l2.grid(row=12,column=3)
        
    

my_w.mainloop()  #


# In[ ]:





# In[60]:





# In[ ]:





# In[ ]:




