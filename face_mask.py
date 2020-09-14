import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import glob
from PIL import Image

def get_data(location='train'):
    folder_location = Path.cwd()
    path = str(folder_location)+'/face_mask_detection/'
    image_path = path+location
    images = []
    
    if location == 'train':
        path_labels = path+'Training_set_face_mask.csv'
        labels = pd.read_csv(path_labels)
        labels = labels['label'].tolist()

        image_path = Path(image_path)
        images = list(image_path.glob('*.jpg'))
        return images, labels
    else:
        image_path = Path(image_path)
        images = list(image_path.glob('*.jpg'))
        return images 
    

class_names = ['No Face Mask', 'Face Mask']

train_images, train_labels = get_data('train')
test_images = get_data('test')

im = Image.open(train_images[0])
im.show()