import cv2
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from keras import datasets, layers, models


(train_images, train_label),(test_images, test_labels)= datasets.cifar10.load_data()
train_images, test_images = train_images/255 , test_images/255

class_name =['Plane','Car','Bird',' Cat','Deer','Dog','Frog','Horse','Ship','Truck']

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_name[train_label[i][0]])
    
plt.show()    



model= models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])

model.fit(train_images,train_label, epochs=10, validation_data=(test_images,test_labels))

loss, accuracy = model.evaluate(test_images,test_labels)

print(f"Loss={loss}")
print(f"Accuracy={accuracy}")

model.save('image_classifier.model.keras')


              