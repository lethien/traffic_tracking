import io
import numpy as np
import tensorflow as tf
import os
import cv2
import json
import pathlib


def update_track_dict(track_dict, trackers, frame_id, obj_class):
    for left, top, right, bottom, track_id in trackers:
        track_id = int(track_id)
        if track_id not in track_dict.keys():
            track_dict[track_id] = {'class': obj_class, 'path': [(left, top, right, bottom, frame_id)]}
        else:
            track_dict[track_id]['path'].append((left, top, right, bottom, frame_id))


def get_videos(input_dir):
    video_paths = []
    for r, d, f in os.walk(input_dir):
        for file in f:
            if '.mp4' in file:
                video_paths.append(os.path.join(r, file))
    
    return video_paths

def extract_frames_from_video(video_path, output_dir, time_stride = 1):    
    video_frame_dir_path = os.path.join(output_dir, os.path.splitext(os.path.basename(video_path))[0])

    vid_cap = cv2.VideoCapture(video_path)
    num_frms, original_fps = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)), vid_cap.get(cv2.CAP_PROP_FPS)

    if not os.path.isdir(video_frame_dir_path):
        os.makedirs(video_frame_dir_path)
    else:
        frames_files = os.listdir(video_frame_dir_path)
        number_files = len(frames_files)
        if number_files != int(num_frms / time_stride):
            for file_object in frames_files:
                file_object_path = os.path.join(video_frame_dir_path, file_object)
                os.unlink(file_object_path)

            for frm_id in tqdm(range(0, num_frms, time_stride)):
                vid_cap.set(cv2.CAP_PROP_POS_FRAMES, frm_id)
                _, im = vid_cap.read()
                frame_img = os.path.join(video_frame_dir_path, str(frm_id) + '.jpg')
                cv2.imwrite(frame_img, im)
    
    vid_cap.release()
    
    frame_files = []
    
    for dirpath,_,filenames in os.walk(video_frame_dir_path):
        for f in filenames:
            frame_files.append(os.path.abspath(os.path.join(video_frame_dir_path, f)))
    
    frame_files.sort(key=os.path.getctime)
    
    return frame_files

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
