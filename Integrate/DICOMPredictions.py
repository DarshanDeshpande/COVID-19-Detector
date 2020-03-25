from pydicom import dcmread
import tensorflow as tf
import numpy as np
import cv2,glob
import json


def create_json(prediction,start,end,sopid):
    datadict = {"protocol_version":"1.0",
            "bounding_boxes_2d": []
            }
    for i in range(len(prediction)):
        datadict['bounding_boxes_2d'].append({"label": prediction[i],
            "SOPInstanceUID": sopid[i],
            "top_left": start[i],
            "bottom_right": end[i]})
    return json.dumps(datadict)


def predict(fpaths):
    predictions,sopid,top_left,bottom_right = [],[],[],[]
    for i,fpath in enumerate(fpaths):
        dataset = dcmread(fpath)
        print(dataset)
        sopid.append(dataset.SOPInstanceUID)
        img = dataset.pixel_array
        image = cv2.resize(img,(400,300))
        image = image*(1./255)
        class_list = ['COVID-19','Non-COVID-19']
        model = tf.keras.models.load_model(r'COVID-19-Detector\Binary\BinaryCOVID-19Classifier.h5')
        pred = model.predict_classes(np.expand_dims(np.expand_dims(image,axis=0),axis=-1))
        predictions.append(class_list[pred[0]])
        top_left.append([0,0])
        bottom_right.append([img.shape[0],img.shape[1]])
    json_data = create_json(predictions,top_left,bottom_right,sopid)
    return json_data


if __name__ == "__main__":
    directory = input("Enter directory: ")
    fpaths = glob.glob(directory+'\\*.dcm')
    json_data = predict(fpaths)
