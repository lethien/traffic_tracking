import numpy as np

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

def get_motion_vector_of_trackers(track_dict):
    motion_vector_list = []
    for tracker_id, tracker_list in track_dict.items():
        if len(tracker_list) > 1:
            last = tracker_list[-1]
            obj_cls = get_obj_class_4types(last[4])
            if obj_cls is not None:    
                last_frame_id = last[5]
                motion_vector_list.append((tracker_list, obj_cls, last_frame_id))
    return motion_vector_list

def counting_moi(video_path, mois, track_dict, similarity_fn):
    """
    Args:
    paths: List of MOI - (first_point, last_point)
    track_dict: Dictionary of track_id: [((bbox), object_class, frame_id)] 

    Returns:
    A list of tuples (frame_id, movement_id, vehicle_class_id)
    """    
    motion_vector_list = get_motion_vector_of_trackers(track_dict)
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