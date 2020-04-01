import tensorflow as tf
import numpy as np
import os,glob
from collections import Counter
import imutils
import cv2
from matplotlib.cm import viridis
from GradientVisualiser import GradCAM


def predict(file_path,choice,model):
    print("Getting you the results for {} classification. Hold on...\n".format(choice))
    images,links=[],[]
    if choice.lower()=='binary':
        class_list = ['Negative','Positive']
        for i in glob.glob(file_path+'/*'):
          img = open(i,'rb').read()
          img = tf.io.decode_jpeg(img,channels=1)
          img = tf.cast(img , tf.float32) * (1. / 255)
          img = tf.image.resize(img,(300,400))
          links.append(i)
          pred = model.predict(np.expand_dims(img.numpy(),axis=0))
          if pred[0] < 0.35:
              images.append('Negative')
          else:
              images.append('Positive')
        for i,j in zip(images,links):
            print(os.path.basename(j).split(r'\\'),i)
        count = Counter(images)
        print("Total Count: {} cases Positive, {} cases Negative".format(count['Positive'],count['Negative']))
    else:
        class_list = ['COVID-19','Normal','BacterialPneumonia']
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
        print("Total Count: {} cases Normal, {} cases detected with COVID-19, {} cases detected with Bacterial Pneumonia".format(count['Normal'],count['COVID-19'],count['BacterialPneumonia']))
    return images,links


def display(images,links,model):
  for i,j in zip(images,links):
      img = open(j,'rb').read()
      image = tf.image.decode_jpeg(img,channels=1)
      img = tf.cast(image,tf.float32) *(1./255)
      img = tf.image.resize(img,(300,400))
      img1 = tf.expand_dims(img,axis=0)
      gc = GradCAM(model,0,layerName='max_pooling2d_19')

      heatmap = gc.compute_heatmap(img1)
      heatmap = cv2.resize(heatmap,(400,300))

      (heatmap, output) = gc.overlay_heatmap(heatmap,image, alpha=0.5)
      image = cv2.resize(cv2.cvtColor(np.array(image),cv2.COLOR_GRAY2RGB),(400,300))
      output1 = np.vstack([image, heatmap, output])
      output1 = imutils.resize(output, height=700)
      cv2.startWindowThread()
      cv2.imshow(j.split('\\')[-1],output)
      cv2.waitKey(0)
      cv2.destroyAllWindows()


if __name__=="__main__":
    choice = str(input("Enter whether you want to do Binary or Multi-Label Classification: ")).capitalize()
    model_path = glob.glob(os.path.join(os.path.dirname(os.path.realpath(__file__)),str(choice))+'\\*.h5')
    path = input("Enter path containing X-ray files: ")
    model = tf.keras.models.load_model(model_path[0])
    images,links = predict(path,choice,model)
    visualise_choice = str(input("Do you want to visualise? (Y/N)")).lower()
    print("Press ESC to change images")
    if visualise_choice == 'y':
        display(images,links,model)
    else:
        exit(0)
