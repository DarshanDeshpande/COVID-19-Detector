"""
Demo script that starts a server which exposes liver segmentation.
Based off of https://github.com/morpheus-med/vision/blob/master/ml/experimental/research/prod/model_gateway/ucsd_server.py
"""

import logging.config
import os
import numpy
import pydicom
import tensorflow as tf
import yaml
from utils import tagged_logger

# ensure logging is configured before flask is initialized


with open('logging.yaml', 'r') as f:
    conf = yaml.safe_load(f.read())
    logging.config.dictConfig(conf)

logger = logging.getLogger('inference')

# pylint: disable=import-error,no-name-in-module
from gateway import Gateway


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
    filename = os.path.join(dirname, 'BinaryClassifier.h5')

    model = tf.keras.models.load_model(filename)

    response_json = {
            'protocol_version': '1.0',
            'parts': [],
            'bounding_boxes_2d': []
        }

    for dicom_instance in dicom_instances:
        dcm = pydicom.read_file(dicom_instance)
        img = dcm.pixel_array
        image = tf.expand_dims(img,axis=-1)
        image = tf.cast(image,tf.float32)*(1./255)
        image = tf.image.resize(image,(400,500))
        image = tf.expand_dims(image,axis=0)
        pred = model.predict(image)
        if pred[0] < 0.35:
            label = 'Negative'
        else:
            label = 'Positive'

        response_json['bounding_boxes_2d'].append({"label": label,
            "SOPInstanceUID": dcm.SOPInstanceUID,
            "top_left": [0,0],
            "bottom_right": [img.shape[0],img.shape[1]]})

    return response_json, []


def request_handler(json_input, dicom_instances, input_digest):
    """
    A mock inference model that returns a mask array of ones of size (height * depth, width)
    """
    transaction_logger = tagged_logger.TaggedLogger(logger)
    transaction_logger.add_tags({ 'input_hash': input_digest })
    transaction_logger.info('mock_model received json_input={}'.format(json_input))

    if json_input['inference_command'] == 'get-bounding-box-2d':
        return get_bounding_box_2d_response(json_input, dicom_instances)
    else:
        return get_empty_response()


if __name__ == '__main__':
    app = Gateway(__name__)
    app.register_error_handler(Exception, handle_exception)
    app.add_inference_route('/', request_handler)

    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=True)
