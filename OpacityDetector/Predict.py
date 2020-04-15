import tensorflow as tf
import numpy as np
import cv2,glob,os
from collections import Counter


def predict(model_path,file_path):
    predictions = []
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully")
    for i in glob.glob(file_path+'\\*'):
        if i.split('\\')[-1].split('.')[-1]=='jfif':
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
    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'OpacityDetector.h5')
    if not os.path.exists(directory):
        raise FileNotFoundError
    else:
        predict(model_path,directory)


