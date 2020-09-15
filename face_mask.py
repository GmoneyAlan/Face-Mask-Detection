import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import glob
from PIL import Image
import cv2

folder_location = Path.cwd()
path = str(folder_location)+'/face_mask_detection/'
image_path_train = path+'train/'
image_path_test = path+'test/'
    
image_height = 128
image_width = 128
class_names = ['No Mask', 'Mask']
#train_ds, train_labels, test_ds = get_data()

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    image_path_train,
    labels='inferred',
    validation_split=0.1,
    class_names=class_names,
    seed=123,
    subset='training',
    image_size=(image_height,image_width),
    batch_size=32,
    shuffle=False
    )

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    image_path_train,
    labels='inferred',
    class_names=class_names,
    validation_split=0.1,
    seed=123,
    subset='training',
    image_size=(image_height,image_width),
    batch_size=32,
    shuffle=False
    )


model = tf.keras.Sequential(
    layers=[
        keras.layers.Flatten(input_shape=(128,128)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(2, activation='relu')        
    ]
)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Make code to divide train folder into 2 categories to better seperate the tensorflow data


print(train_ds.class_names)
plt.figure(figsize=(10,10))
for image, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(image[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
plt.show()

#model.fit(train_ds, epochs=10, verbose=2)






