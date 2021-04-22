import sys
import numpy as np

from tensorflow import keras
from PIL import Image

path = './data/external/'

def main():
    try:
        argv = sys.argv[1:]

        cnn_call = argv[0]
        image_call = argv[1]
        
        if cnn_call == 'vgg16':
            predict_vgg16(image_call)
        elif cnn_call == 'cnn' :
            predict_vgg16(image_call)
        else:
            raise Exception("Error: Check the arguments passe") 
    except :
        print ('Error: Check the arguments passed')

def image_treament_vgg16 (image):

    image = Image.open(path+image).resize((224, 224), Image.ANTIALIAS)
    image = np.asarray(image)/ 255 
    image = image.reshape(1,224,224,3)
    
    return image


def image_treament_cnn_simple (image):

    image = Image.open(path+image).convert('LA').resize((50, 50), Image.ANTIALIAS)
    image = np.asarray(image)/ 255 
    image = image.reshape(1,50,50,2)
    
    return image

def predict_vgg16(image_call):
    
    image_process = image_treament_vgg16(image_call)
    model = keras.models.load_model('./models/model_vgg16_TL')
    predict = model.predict_classes(image_process)

    if predict == 1 :
        print('Quarto sujo')
    else :
        print('Quarto limpo')

def predict_cnn_simple(image_call):

    image_process = image_treament_cnn_simple(image_call)
    model = keras.models.load_model('./models/model_cnn_simple')
    predict = model.predict_classes(image_process)
    
    if predict == 1 :
        print('Quarto sujo')
    else :
        print('Quarto limpo')

if __name__ == '__main__':
    main()


