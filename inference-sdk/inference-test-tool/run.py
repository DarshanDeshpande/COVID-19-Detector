"""
This script is meant to mock an inference request from Arterys to your model.
Before executing this script, you must update mock_server.py#request_handler
or implement your own server to direct the request to your model
function (model_fn) and adjust the request literal below as necessary, then
run the container.

Ex. python mock_upload_study.py <path_to_dicom_dir>
"""

import argparse
import os
import requests
import json

from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
from base64 import b64encode
from requests_toolbelt import MultipartEncoder
from requests_toolbelt.multipart import decoder
import pydicom
import numpy as np
import test_inference_mask

from utils import load_image_data, sort_images

def upload_study_me(file_path,host, port):
    file_dict = []
    headers = {'Content-Type': 'multipart/related; '}
    request_json = {'request': 'post', 
                    'route': '/',
                    'inference_command': 'get-bounding-box-2d'}
    
    images = load_image_data(file_path)
    images = sort_images(images)

    width = 0
    height = 0
    count = 0
    for image in images:
        try:
            dcm_file = pydicom.dcmread(image.path)
            if width == 0 or height == 0:
                width = dcm_file.Columns
                height = dcm_file.Rows
            count += 1
            field = str(count)
            fo = open(image.path, 'rb').read()
            filename = os.path.basename(os.path.normpath(image.path))
            file_dict.append((field, (filename, fo, 'application/dicom')))
        except:
            print('File {} is not a DICOM file'.format(image.path))
            continue
    
    print('Sending {} files...'.format(count))
    request_json['depth'] = count
    request_json['height'] = height
    request_json['width'] = width

    file_dict.insert(0, ('request_json', ('request', json.dumps(request_json).encode('utf-8'), 'text/json')))
    
    me = MultipartEncoder(fields=file_dict)
    boundary = me.content_type.split('boundary=')[1]
    headers['Content-Type'] = headers['Content-Type'] + 'boundary="{}"'.format(boundary)

    r = requests.post('http://' + host + ':' + port + '/', data=me, headers=headers)
    
    if r.status_code != 200:
        print("Got error status code ", r.status_code)
        exit(1)

    multipart_data = decoder.MultipartDecoder.from_response(r)

    json_response = json.loads(multipart_data.parts[0].text)
    print("JSON response:", json_response)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="Path to dicom directory to upload.")
    parser.add_argument("--host", default='arterys-inference-sdk-server', help="Host where inference SDK is hosted")
    parser.add_argument("-p", "--port", default='8000', help="Port of inference SDK host")
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = parse_args()
    upload_study_me(args.file_path,args.host, args.port)
