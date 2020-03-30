import tensorflow as tf
import numpy as np
import os,glob
from collections import Counter


def preprocess(file_path,choice,model_path):
    print("Getting you the results. Hold on...\n")
    images,links=[],[]
    if choice.lower()=='binary':
        class_list = ['Positive','Negative']
        model = tf.keras.models.load_model(model_path)
    else:
        class_list = ['COVID-19','Normal','BacterialPneumonia']
        model = tf.keras.models.load_model(model_path)
    print("Model loaded. Running Inference now...\n")
    for i in glob.glob(file_path+'/*'):
      img = open(i,'rb').read()
      img = tf.io.decode_jpeg(img,channels=1)
      img = tf.cast(img , tf.float32) * (1. / 255)
      img = tf.image.resize(img,(300,400))
      links.append(i)
      images.append(np.argmax(model.predict(np.expand_dims(img.numpy(),axis=0))))
    print(images)
    images = [class_list[i] for i in images]
    for i,j in zip(images,links):
        print(os.path.basename(j).split(r'\\'),i)
    count = Counter(images)
    if choice.lower()=='binary':
        print("Total Count: {} cases Positive, {} cases Negative".format(count['Positive'],count['Negative']))
    else:
        print("Total Count: {} cases Normal, {} cases detected with COVID-19, {} cases detected with Bacterial Pneumonia".format(count['Normal'],count['COVID-19'],count['BacterialPneumonia']))


if __name__=="__main__":
    choice = str(input("Enter whether you want to do Binary or Multi-Label Classification: "))
    model_path = glob.glob(os.path.join(os.path.dirname(os.path.realpath(__file__)),str(choice))+'\\*.h5')
    path = input("Enter path containing X-ray files: ")
    preprocess(path,choice,model_path[0])
