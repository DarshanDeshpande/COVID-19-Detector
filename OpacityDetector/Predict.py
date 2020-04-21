import glob
import os
import numpy as np
import zipfile
from collections import Counter
import tensorflow as tf
from GradientVisualiser import GradCAM
import cv2, imutils


def predict(model, file_path, choice):
    images,predictions = [],[]
    if choice == 'resnet':
        dim, channels = (200, 200), 3
    else:
        dim, channels = (400, 400), 1
    for i in glob.glob(file_path + '\\*'):
        if i.split('\\')[-1].split('.')[-1] == 'jfif':
            print(
                "Invalid file format '.jfif' for file: {}. Supported formats: jpeg,png,jpg. Skipping current file".format(
                    i))
            continue
        images.append(i)
        img = open(i, 'rb').read()
        img = tf.image.decode_jpeg(img, channels=channels)
        img = tf.image.resize(img, dim)
        img = img * (1. / 255)
        pred = model.predict(tf.expand_dims(img, axis=0))
        pred = "Negative" if pred[0] < 0.3 else "Positive"
        print(i, '-->', pred)
        predictions.append(pred)
    print(Counter(predictions))
    return images


def display(links, model, category, layer_name):
    if 'resnet' in category.lower():
        dim, channels = (200, 200), 3
    else:
        dim, channels = (400, 400), 1
    for j in links:
        img = open(j, 'rb').read()
        image = tf.image.decode_jpeg(img, channels=channels)
        img = tf.cast(image, tf.float32) * (1. / 255)
        img = tf.image.resize(img, dim)
        img1 = tf.expand_dims(img, axis=0)
        gc = GradCAM(model, 0, layerName=layer_name)

        heatmap = gc.compute_heatmap(img1)
        heatmap = cv2.resize(heatmap, dim)

        (heatmap, output) = gc.overlay_heatmap(heatmap, image, dim=dim, channels=channels, alpha=0.5)
        if channels==1:
            image = cv2.resize(cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2RGB), dim)
        else:
            image = cv2.resize(np.array(image),dim)
        output1 = np.vstack([image, output])
        output1 = imutils.resize(output1, height=700)
        cv2.startWindowThread()
        cv2.imshow(j.split('\\')[-1], output1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    directory = str(input("Enter path to the directory which contains the images: "))
    choice = str(input("Which model to use? \n1. COVID-Resnet(Recommended)\n2. Custom\n")).lower()
    if 'resnet' in choice.lower():
        if not os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'SecondPrunedResnet.h5')):
            with zipfile.ZipFile('COVID-Resnet.zip', 'r') as zipObj:
                zipObj.extractall()
                print("Extracted model successfully")
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'SecondPrunedResnet.h5')
    else:
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'OpacityDetector.h5')

    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully")

    if not os.path.exists(directory):
        raise FileNotFoundError
    else:
        links = predict(model, directory, choice.lower())

        for i in reversed(model.layers):
            if len(i.output.shape)==4:
                layer_name = i.name
                break
        visualise_choice = str(input("Do you want to visualise? (Y/N)")).lower()
        if visualise_choice == 'y':
            print("Press any key to view the next image")
            display(links,model,str(choice),layer_name)
        else:
            exit(0)
