import io
import numpy as np
import tensorflow as tf
import os
import cv2
from tqdm import tqdm

import time

from PIL import Image
from six import BytesIO

from video_utils import *
from img_utils import *
from bb_polygon import *

def object_detect_image(image, detect_fn):
    input_tensor = tf.convert_to_tensor(np.expand_dims(image.astype(np.uint8), 0), dtype=tf.float32)

    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    return detections


def video_object_dectection(video_path, detect_fn, category_index, roi, 
                                         video_output_dir, output_to_video = False,
                                         from_frame = 0, to_frame = None, time_stride = 1):       
    vid_cap = cv2.VideoCapture(video_path)
    num_frms, original_fps = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)), vid_cap.get(cv2.CAP_PROP_FPS)

    vid_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, img = vid_cap.read()
    height, width, layers = img.shape
    size = (width,height)
    
    if output_to_video:
        output_file = os.path.join(video_output_dir, os.path.splitext(os.path.basename(video_path))[0] + '.mp4')
        if os.path.exists(output_file):
            os.remove(output_file)
        out = cv2.VideoWriter(output_file,cv2.VideoWriter_fourcc(*'DIVX'), int(original_fps/time_stride), size)
    
    frames_to_look = range(min(from_frame, num_frms), 
                           min(to_frame, num_frms) if to_frame is not None else num_frms, 
                           time_stride)    
    for frame_id in tqdm(frames_to_look):  
        vid_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        _, frame = vid_cap.read()  
        
        # Get object detection bounding boxes
        detections = object_detect_image(frame, detect_fn) 
        
        if output_to_video:
            # draw ROI and bounding boxes onto frame
            image_np_with_detections = draw_roi_on_image(frame,roi)
        
            image_np_with_detections = draw_boxes(image_np_with_detections, detections['detection_boxes'], 
                                                  [category_index[i+1]['name'] for i in detections['detection_classes']], detections['detection_scores'],
                                                 max_boxes=100, min_score=0.3)
            out.write(image_np_with_detections)
    
    vid_cap.release()
    if output_to_video:
        out.release()


def video_object_dectection_and_tracking(video_path, detect_fn, tracker, category_index, roi, 
                                         video_output_dir, output_to_video = False,
                                         from_frame = 0, to_frame = None, time_stride = 1, min_score=0.3):   
    track_dict = {}

    vid_cap = cv2.VideoCapture(video_path)
    num_frms, original_fps = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)), vid_cap.get(cv2.CAP_PROP_FPS)

    vid_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, img = vid_cap.read()
    height, width, layers = img.shape
    size = (width,height)
    
    if output_to_video:
        output_file = os.path.join(video_output_dir, os.path.splitext(os.path.basename(video_path))[0] + '_with_tracking.mp4')
        if os.path.exists(output_file):
            os.remove(output_file)
        out = cv2.VideoWriter(output_file,cv2.VideoWriter_fourcc(*'DIVX'), int(original_fps/time_stride), size)
    
    frames_to_look = range(min(from_frame, num_frms), 
                           min(to_frame, num_frms) if to_frame is not None else num_frms, 
                           time_stride)    
    for frame_id in tqdm(frames_to_look):    
        vid_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        _, frame = vid_cap.read()
        
        # Get object detection bounding boxes
        detections = object_detect_image(frame, detect_fn) 
       
        # update SORT trackers         
        dets = change_detections_to_image_coordinates(detections, roi, width, height, min_score)
        tracked_objects = []
        if len(dets) > 0:
            tracked_objects = tracker.update(dets)
            update_track_dict(track_dict, tracked_objects, frame_id) 

        if output_to_video: 
            image_np_with_detections = draw_roi_on_image(frame,roi)
            image_np_with_detections = draw_boxes_and_lines(image_np_with_detections, tracked_objects, track_dict, category_index)
            out.write(image_np_with_detections[:, :, ::-1])                
    
    vid_cap.release()
    if output_to_video:
        out.release()
    
    return track_dict

def update_track_dict(track_dict, trackers, frame_id):
    for left, top, right, bottom, track_id, obj_class in trackers:
        track_id = int(track_id)
        if track_id not in track_dict.keys():
            track_dict[track_id] = [(left, top, right, bottom, obj_class, frame_id)]
        else:
            track_dict[track_id].append((left, top, right, bottom, obj_class, frame_id))
