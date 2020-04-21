import glob
import os
import zipfile
from collections import Counter
import tensorflow as tf


def predict(model_path,file_path,choice):
    predictions = []
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully")
    if choice == 'resnet':
        for i in glob.glob(file_path+'\\*'):
            if i.split('\\')[-1].split('.')[-1]=='jfif':
                print("Invalid file format '.jfif' for file: {}. Supported formats: jpeg,png,jpg. Skipping current file".format(i))
                continue
            img = open(i,'rb').read()
            img = tf.image.decode_jpeg(img,channels=3)
            img = tf.image.resize(img,(200,200))
            img = img*(1./255)
            pred = model.predict(tf.expand_dims(img,axis=0))
            pred = "Negative" if pred[0]<0.3 else "Positive"
            print(i,'-->',pred)
            predictions.append(pred)
        print(Counter(predictions))
    else:
        for i in glob.glob(file_path+'\\*'):
            if i.split('\\')[-1].split('.')[-1]=='jfif':
                print("Invalid file format '.jfif' for file: {}. Supported formats: jpeg,png,jpg. Skipping current file".format(i))
                continue
            img = open(i,'rb').read()
            img = tf.image.decode_jpeg(img,channels=1)
            img = tf.image.resize(img,(400,400))
            img = img*(1./255)
            pred = model.predict(tf.expand_dims(img,axis=0))
            pred = "Negative" if pred[0]<0.3 else "Positive"
            print(i,'-->',pred)
            predictions.append(pred)
        print(Counter(predictions))


if __name__=='__main__':
    directory = str(input("Enter path to the directory which contains the images: "))
    choice = str(input("Which model to use? \n1. Resnet(Recommended)\n2. Custom\n"))
    if choice.lower()=='resnet':
        if not os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)),'SecondPrunedResnet.h5')):
            with zipfile.ZipFile('COVID-Resnet.zip', 'r') as zipObj:
                zipObj.extractall()
                print("Extracted model successfully")
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'SecondPrunedResnet.h5')
    else:
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'OpacityDetector.h5')

    if not os.path.exists(directory):
        raise FileNotFoundError
    else:
        predict(model_path,directory,choice.lower())


