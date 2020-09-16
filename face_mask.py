import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

folder_location = Path.cwd()
path = str(folder_location)+'/face_mask_detection/'
image_path_train = path+'train/'
image_path_test = path+'test/'
    
image_height = 128
image_width = 128
class_names = ['No Mask', 'Mask']

'''
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    image_path_train,
    labels='inferred',
    validation_split=0.1,
    class_names=class_names,
    seed=123,
    subset='training',
    image_size=(image_height,image_width),
    batch_size=32,
    shuffle=True
    )

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    image_path_train,
    labels='inferred',
    class_names=class_names,
    validation_split=0.1,
    seed=123,
    subset='validation',
    image_size=(image_height,image_width),
    batch_size=32,
    shuffle=True
    )


#Load Images into cache, so after the first epoch the model should run faster

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


model = tf.keras.Sequential(
    layers=[
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(image_height, image_width, 3)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(len(class_names))        
    ]
)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


plt.figure(figsize=(10,10))

for image, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(image[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
plt.show()

print(model.summary())

model.fit(train_ds,validation_data=val_ds, epochs=3, verbose=2)
test_loss, test_acc = model.evaluate(train_ds, verbose=2)
print('loss:', test_loss, '  accuracy:', test_acc)
model.save("model.h5")
'''

model = tf.keras.models.load_model('model.h5')

prediction_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])
test_images = []
for i in range(1,11):
    test = (tf.keras.preprocessing.image.load_img(
        image_path_test+str('Image_'+str(i)+'.jpg'),
        target_size=(image_height,image_width)
    ))
    test_arr = tf.keras.preprocessing.image.img_to_array(test)
    test_images.append(test_arr)

test_images = np.asarray(test_images)
print(test_images.shape)

prediction = prediction_model.predict(test_images)

for i in range(test_images.shape):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(test_images[i].astype('uint8'))
        plt.title(class_names[np.argmax(prediction[i])])
        plt.axis('off')
plt.show()

