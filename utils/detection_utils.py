import io
import numpy as np
import tensorflow as tf
import os
import cv2
from tqdm import tqdm

from PIL import Image
from six import BytesIO

from video_utils import *
from img_utils import *
from bb_polygon import *

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def object_detect_image(image_path, detect_fn):
    image_np = load_image_into_numpy_array(image_path)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    return image_np, detections


def video_object_dectection(video_path, detect_fn, category_index,
                                         frame_output_dir, zone_info_dir, 
                                         video_output_dir, output_to_video = False,
                                         from_frame = 0, to_frame = None, time_stride = 1):   
    extracted_frames = extract_frames_from_video(video_path, frame_output_dir)
    roi, mois = extract_video_info(video_path, zone_info_dir)
    
    img = cv2.imread(extracted_frames[0])
    height, width, layers = img.shape
    size = (width,height)
    
    if output_to_video:
        output_file = os.path.join(video_output_dir, os.path.splitext(os.path.basename(video_path))[0] + '.mp4')
        out = cv2.VideoWriter(output_file,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
    frames_to_look = range(min(from_frame, len(extracted_frames)), 
                           min(to_frame, len(extracted_frames)) if to_frame is not None else len(extracted_frames), 
                           time_stride)    
    for frame_id in tqdm(frames_to_look):    
        frame = extracted_frames[frame_id]
        
        # Get object detection bounding boxes
        frame_img, detections = object_detect_image(frame, detect_fn) 
        
        if output_to_video:
            # draw ROI and bounding boxes onto frame
            image_np_with_detections = draw_roi_on_image(frame_img,roi)
        
            image_np_with_detections = draw_boxes(image_np_with_detections, detections['detection_boxes'], 
                                                  [category_index[i+1]['name'] for i in detections['detection_classes']], detections['detection_scores'],
                                                 max_boxes=100, min_score=0.3)
            out.write(image_np_with_detections[:, :, ::-1])
    
    if output_to_video:
        out.release()


def video_object_dectection_and_tracking(video_path, detect_fn, tracker, category_index,
                                         frame_output_dir, zone_info_dir, 
                                         video_output_dir, output_to_video = False,
                                         from_frame = 0, to_frame = None, time_stride = 1):   
    track_dict = {}

    extracted_frames = extract_frames_from_video(video_path, frame_output_dir)
    roi, mois = extract_video_info(video_path, zone_info_dir)
    
    img = cv2.imread(extracted_frames[0])
    height, width, layers = img.shape
    size = (width,height)
    
    if output_to_video:
        output_file = os.path.join(video_output_dir, os.path.splitext(os.path.basename(video_path))[0] + '_with_tracking.mp4')
        out = cv2.VideoWriter(output_file,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
    frames_to_look = range(min(from_frame, len(extracted_frames)), 
                           min(to_frame, len(extracted_frames)) if to_frame is not None else len(extracted_frames), 
                           time_stride)    
    for frame_id in tqdm(frames_to_look):    
        frame = extracted_frames[frame_id]
        
        # Get object detection bounding boxes
        frame_img, detections = object_detect_image(frame, detect_fn) 
        
        if output_to_video:
            # draw ROI and bounding boxes onto frame
            image_np_with_detections = draw_roi_on_image(frame_img,roi)
        
        # update SORT trackers         
        dets = change_detections_to_image_coordinates(detections, roi, width, height, min_score=0.2)
        if len(dets) > 0:
            tracked_objects = tracker.update(dets)
            update_track_dict(track_dict, tracked_objects, frame_id)
            if output_to_video:
                image_np_with_detections = draw_boxes_and_lines(image_np_with_detections, tracked_objects, track_dict, category_index)
                out.write(image_np_with_detections[:, :, ::-1])
    
    if output_to_video:
        out.release()
    
    return track_dict