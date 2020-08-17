import io
import numpy as np
import tensorflow as tf
import os
import cv2
import json
import pathlib
from tqdm import tqdm

def get_videos(input_dir):
    video_paths = []
    for r, d, f in os.walk(input_dir):
        for file in f:
            if '.mp4' in file:
                video_paths.append(os.path.join(r, file))
    
    return video_paths

def extract_video_info(video_path, info_dir):
    video_frame_dir_path = os.path.join(info_dir, os.path.splitext(os.path.basename(video_path))[0] + ".json")
    with open(video_frame_dir_path) as jsonfile:
        jf_content = json.load(jsonfile)

        roi = None
        mois = {}

        for shape in jf_content['shapes']:
            if shape["shape_type"] == "polygon":
                roi = [(int(x), int(y)) for x, y in shape['points']]
            elif shape["shape_type"] == "line":
                moi_id = str(int(shape['label'][-2:]))
                mois[moi_id] = [(int(x), int(y)) for x, y in shape['points']]
                
    return roi, mois

# Download labels file
def download_labels(filename):
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
    label_dir = tf.keras.utils.get_file(fname=filename,
                                        origin=base_url + filename,
                                        untar=False)
    label_dir = pathlib.Path(label_dir)
    return str(label_dir)

# Download and extract model
def download_model(model_name, model_date):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name,
                                        origin=base_url + model_date + '/' + model_file,
                                        untar=True)
    return str(model_dir)
