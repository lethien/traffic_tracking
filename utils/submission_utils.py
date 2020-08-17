import os

def write_tracking_result(video_path, result_output_dir, moi_detection_list):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    result_filename = os.path.join(result_output_dir, 'videos_mois_counting', video_id + '_counting.txt')
    with open(result_filename, 'w') as result_file:
        for frame_id, movement_id, vehicle_class_id in moi_detection_list:
            result_file.write('{} {} {} {}\n'.format(video_id, frame_id, movement_id, vehicle_class_id))


def write_submission(result_output_dir):
    result_filename = os.path.join(result_output_dir, 'combined_submission.txt')
    mois_output_dir = os.path.join(result_output_dir, 'videos_mois_counting')    
    with open(result_filename, 'w') as result_file:        
        for r, d, f in os.walk(mois_output_dir):            
            for file in f:
                with open(os.path.join(r, file)) as in_f:
                    for line in in_f:
                        result_file.write(line)
