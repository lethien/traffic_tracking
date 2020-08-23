import numpy as np
from bb_polygon import *

def get_obj_class_4types(obj_cls):
    if obj_cls == 1 or obj_cls == 2 or obj_cls == 4: # two wheels
        return 1
    elif obj_cls == 3: # small four wheels
        return 2
    elif obj_cls == 6: # medium four wheels
        return 3
    elif obj_cls == 7 or obj_cls == 8: # big four wheels
        return 4
    else: 
        return None

def get_tracker_list_completed(tracker_list, roi):
    average_time_stride = int(np.mean(np.diff(np.array(tracker_list)[:,5])))

    last_frame = tracker_list[-1]
    last_frame_id = last_frame[5]

    centers_x = (np.array(tracker_list)[:,2] + np.array(tracker_list)[:,0]) / 2.0
    centers_y = (np.array(tracker_list)[:,3] + np.array(tracker_list)[:,1]) / 2.0

    average_x_increase = np.mean(np.diff(centers_x))
    average_y_increase = np.mean(np.diff(centers_y))

    average_x_increase_per_frame = (np.sign(average_x_increase) * np.max([average_time_stride, np.abs(average_x_increase)])) / average_time_stride
    average_y_increase_per_frame = (np.sign(average_y_increase) * np.max([average_time_stride, np.abs(average_y_increase)])) / average_time_stride

    while True:        
        pred_next_left = last_frame[0] + average_x_increase_per_frame
        pred_next_top = last_frame[1] + average_y_increase_per_frame
        pred_next_right = last_frame[2] + average_x_increase_per_frame
        pred_next_bottom = last_frame[3] + average_y_increase_per_frame
        
        if not check_bbox_intersect_polygon(roi, (pred_next_left, pred_next_top, pred_next_right, pred_next_bottom)):
            break
        else:            
            tracker_list.append((pred_next_left, pred_next_top, pred_next_right, pred_next_bottom, last_frame[4], last_frame_id + 1))
            last_frame = tracker_list[-1]
            last_frame_id = last_frame[5]            
    
    return last_frame_id, tracker_list


def get_motion_vector_of_trackers(track_dict, roi):
    motion_vector_list = []
    for tracker_id, tracker_list in track_dict.items():
        if len(tracker_list) >= 2:  
            cls_vals, cls_counts = np.unique(np.array(tracker_list)[:,4], return_counts=True)
            obj_cls = get_obj_class_4types(cls_vals[np.argmax(cls_counts)])
    
            if obj_cls is not None:  
                last_frame_id, tracker_list_completed = get_tracker_list_completed(tracker_list, roi)                          
                motion_vector_list.append((tracker_list_completed, obj_cls, last_frame_id))

    return motion_vector_list

def counting_moi(video_path, roi, mois, track_dict, similarity_fn):
    """
    Args:
    paths: List of MOI - (first_point, last_point)
    track_dict: Dictionary of track_id: [((bbox), object_class, frame_id)] 

    Returns:
    A list of tuples (frame_id, movement_id, vehicle_class_id)
    """    
    motion_vector_list = get_motion_vector_of_trackers(track_dict, roi)
    moi_detection_list = []
    for motion_vector in motion_vector_list:
        max_similariry = -2
        movement_id = ''
        last_frame = 0
        for movement_label, movement_vector in mois.items():
            similariry = similarity_fn(movement_vector, motion_vector[0])
            if similariry > max_similariry:
                max_similariry = similariry
                movement_id = movement_label
        obj_cls = motion_vector[1]
        last_frame = motion_vector[2]
        moi_detection_list.append((last_frame, int(movement_id), obj_cls))
    
    moi_detection_list = np.array(moi_detection_list)
    moi_detection_list = moi_detection_list[moi_detection_list[:,0].argsort()]
    
    return moi_detection_list