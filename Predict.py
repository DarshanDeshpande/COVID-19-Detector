import tensorflow as tf
import numpy as np
import os,cv2,glob

model = tf.keras.models.load_model('COVID-19-Detector.h5')
class_list = ['Normal','COVID-19','Bacterial Pneumonia']


def preprocess(file_path):
    images,links=[],[]
    for i in glob.glob(file_path+'/*'):
      img = open(i,'rb').read()
      img = tf.io.decode_jpeg(img,channels=1)
      img = tf.cast(img , tf.float32) * (1. / 255)
      img = tf.image.resize(img,(300,400))
      links.append(i)
      images.append(model.predict_classes(np.expand_dims(img.numpy(),axis=0)))
    images = [class_list[i[0]] for i in images]
    for i,j in zip(images,links):
        print(os.path.basename(j).split(r'\\'),i)


if __name__=="__main__":
    path = input("Enter path containing X-ray files: ")
    preprocess(path)
