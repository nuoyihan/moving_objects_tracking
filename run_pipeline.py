from track import detect_and_track_red_dots
from trajectory_vis import visualize_trajectory
from crop import crop_video_using_details, detect_paper_get_crop_details
import os
import pickle
import argparse

parser = argparse.ArgumentParser(description='Process and track a video.')
parser.add_argument('video_name', type=str, help='The name of the video file (without extension)')
args = parser.parse_args()
video_name = f'{args.video_name}.MOV'

input_dir = 'data/video_in/'
output_dir = 'data/video_out/'
path_plot_dir = 'data/tracked_paths_plots/'
path_coords_dir = 'data/tracked_paths_coords/'



output_name = 'tracked_' + video_name
path_plot_name = path_plot_dir  + "tracked_" + os.path.splitext(video_name)[0] + '.png'
path_coords_name = path_coords_dir + "tracked_" + os.path.splitext(video_name)[0] + '.pkl'

input_video_path = input_dir + video_name

cropped_video_path = input_dir + "cropped_" + video_name

output_video_path = output_dir + output_name


_, crop_detail = detect_paper_get_crop_details(input_video_path)
crop_video_using_details(input_video_path, cropped_video_path , crop_detail)
trajectory, width, height = detect_and_track_red_dots(cropped_video_path, output_video_path)

if trajectory is not None:
    visualize_trajectory(trajectory, width, height, path_plot_name)
    with open(path_coords_name, 'wb') as file:
        pickle.dump(trajectory, file)
