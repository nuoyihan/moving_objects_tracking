prepare environment:<br>
conda create --name tracking_env<br>
conda activate tracking_env<br>
conda install numpy matplotlib opencv<br>

--
to run tracking for a video "video.mov"<br>
put "video.mov" in "data/video_in" directory and run this on the command line "python run_pipeline.py video"<br>

--
video with tracked markers annotated will appear in "data/video_out"<br>

a visualization of the tracked path will be in "data/tracked_paths_plots"<br>
coordinates of the tracked path will be in "data/path_plots"<br>
