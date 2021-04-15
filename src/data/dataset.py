# -*- coding: utf-8 -*-
from PIL import Image

import os
import numpy as np

train_data = '../data/raw/train/'
test_data = '../data/raw/test/'
val_data = '../data/raw/val/'

process_train = '../data/processed/train/'
process_test = '../data/processed/test/'
process_val = '../data/processed/val/'

category_names = ['messy/', 'clean/']

def make_dataset_50x50x2():
    
    image_size = 50
    
    # Train GrayScale with new size 50x50
    
    for category in category_names:
        path = train_data+category
        path_processed = process_train+category

        for image in os.listdir(path):

            image_treatment = Image.open(path+image).convert('LA').resize((image_size, image_size), Image.ANTIALIAS)
            image_treatment.save(path_processed+image, 'PNG')
    
    # Test GrayScale with new size 50x50
    
    for image in os.listdir(test_data):

        image_treatment = Image.open(test_data+image).convert('LA').resize((image_size, image_size), Image.ANTIALIAS)
        image_treatment.save(process_test+image, 'PNG')
    
    # Val GrayScale with new size 50x50
    
    for category in category_names:
    
        path = val_data+category
        path_processed = process_val+category

        for image in os.listdir(path):

            image_treatment = Image.open(path+image).convert('LA').resize((image_size, image_size), Image.ANTIALIAS)
            image_treatment.save(path_processed+image, 'PNG')

            
def dataset_train():
    
    clean_train = []
    messy_train = []
    y_train = []

    for category in category_names:

        path_processed = process_train+category

        for image in os.listdir(path_processed):

            image_treatment = Image.open(path_processed+image)

            if category == 'messy/' :
                messy = np.asarray(image_treatment) /255 # bit format
                messy_train.append(messy)
                y_train.append(1)
            else:
                clean = np.asarray(image_treatment)/255 # bit format
                clean_train.append(clean)
                y_train.append(0)
    
    x_train = np.concatenate((messy_train,clean_train),axis=0)
    y_train = np.array(y_train)
    
    return (x_train, y_train)
    
def dataset_val():
    
    x_val = []
    y_val = []

    for category in category_names:

        path_processed = process_val+category

        for image in os.listdir(path_processed):

            image_treatment = Image.open(path_processed+image).resize((50,50))

            if category == 'messy/' :
                messy = np.asarray(image_treatment) /255 # bit format
                x_val.append(messy)
                y_val.append(1)
            else:
                clean = np.asarray(image_treatment)/255 # bit format
                x_val.append(clean)
                y_val.append(0)
    
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    
    return (x_val, y_val)    
