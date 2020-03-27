"""
Demo script that starts a server which exposes liver segmentation.

Based off of https://github.com/morpheus-med/vision/blob/master/ml/experimental/research/prod/model_gateway/ucsd_server.py
"""

import functools
import logging
import logging.config
import os
import tempfile
import yaml
import json
import numpy
import pydicom
import tensorflow as tf
from utils.image_conversion import convert_to_nifti
from utils import tagged_logger

# ensure logging is configured before flask is initialized

with open('logging.yaml', 'r') as f:
    conf = yaml.safe_load(f.read())
    logging.config.dictConfig(conf)

logger = logging.getLogger('inference')

# pylint: disable=import-error,no-name-in-module
from gateway import Gateway
from flask import make_response

def handle_exception(e):
    logger.exception('internal server error %s', e)
    return 'internal server error', 500

def get_empty_response():
    response_json = {
        'protocol_version': '1.0',
        'parts': []
    }
    return response_json, []

def get_bounding_box_2d_response(json_input, dicom_instances):

    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'BinaryCOVID-19Classifier.h5')

    class_list = ['COVID-19','Other']
    model = tf.keras.models.load_model(filename)
    prediction,sopid,top_left,bottom_right = [],[],[],[]

    response_json = {
            'protocol_version': '1.0',
            'parts': [],
            'bounding_boxes_2d': []
        }

    for dicom_instance in dicom_instances:
        dcm = pydicom.read_file(dicom_instance)
        img = dcm.pixel_array
        image = numpy.expand_dims(numpy.expand_dims(img,axis=0),axis=-1)

        pred = model.predict_classes(image)
        pred = class_list[pred[0]]
        prediction.append(pred)
        sopid.append(dcm.SOPInstanceID)
        top_left.append([0,0])
        bottom_right.append([img.shape[0],img.shape[1]])

    for i in range(len(prediction)):
        response_json['bounding_boxes_2d'].append({"label": prediction[i],
            "SOPInstanceUID": sopid[i],
            "top_left": top_left[i],
            "bottom_right": bottom_right[i]})

    return response_json, []

def get_probability_mask_response(json_input, dicom_fpath_list):
    response_json = {
            'protocol_version': '1.0',
            'parts': [
                {
                    'label': 'Mock seg',
                    'binary_type': 'probability_mask',
                    'binary_data_shape': {
                        'timepoints': 1,
                        'depth': json_input['depth'],
                        'width': json_input['width'],
                        'height': json_input['height']
                    }
                }
            ]
    }

    array_shape = (json_input['depth'], json_input['height'], json_input['width'])
    
    # This code produces a mask that grows from the center of the image outwards as the image slices advance
    mask = numpy.zeros(array_shape, dtype=numpy.uint8)
    mid_x = int(json_input['width'] / 2)
    mid_y = int(json_input['height'] / 2)
    for s in range(json_input['depth']):
        offset_x = int(s / json_input['depth'] * mid_x)
        offset_y = int(s / json_input['depth'] * mid_y)
        indices = numpy.ogrid[mid_y - offset_y : mid_y + offset_y, mid_x - offset_x : mid_x + offset_x]
        mask[s][tuple(indices)] = 255

    return response_json, [mask]


def request_handler(json_input, dicom_instances, input_digest):
    """
    A mock inference model that returns a mask array of ones of size (height * depth, width)
    """
    transaction_logger = tagged_logger.TaggedLogger(logger)
    transaction_logger.add_tags({ 'input_hash': input_digest })
    transaction_logger.info('mock_model received json_input={}'.format(json_input))

    # If your model accepts Nifti files as input then uncomment the following lines:
    # convert_to_nifti(dicom_instances, 'nifti_output.nii')
    # print("Converted file to nifti 'nifti_output.nii'")
    
    if json_input['inference_command'] == 'get-bounding-box-2d':
        return get_bounding_box_2d_response(json_input, dicom_instances)
    elif json_input['inference_command'] == 'get-probability-mask':
        return get_probability_mask_response(json_input, dicom_instances)
    else:
        return get_empty_response()


if __name__ == '__main__':
    app = Gateway(__name__)
    app.register_error_handler(Exception, handle_exception)
    app.add_inference_route('/', request_handler)

    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=True)
