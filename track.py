import cv2
import numpy as np
from preprocess_video import apply_gaussian_blur, enhance_contrast, apply_morphological_operations

# Parameters
PERSISTENCE_THRESHOLD = 3  # Number of frames a dot needs to be consistently detected
MAX_DISTANCE_CHANGE = 20  # Maximum allowed change in distance between dots to consider them as the same group

def detect_and_track_red_dots(video_path, output_video_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    lower_red = np.array([0, 60, 40])
    upper_red = np.array([10, 255, 255])
    lower_red2 = np.array([170, 60, 40])
    upper_red2 = np.array([180, 255, 255])

    trajectory = []
    tracked_positions = None  # To store the initial three red dots

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error in frame capture.")
            break

        # Apply preprocessing steps
        frame = enhance_contrast(frame)
        frame = apply_gaussian_blur(frame)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Apply morphological operations to clean up the mask
        mask = apply_morphological_operations(mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_positions = []
        for contour in contours:
            if cv2.contourArea(contour) > 10:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                detected_positions.append(center)
                cv2.circle(frame, center, int(radius), (0, 255, 0), 2)

        if tracked_positions is None:
            # First frame: Initialize by selecting the three closest red dots
            if len(detected_positions) >= 3:
                # Calculate pairwise distances
                distances = []
                for i in range(len(detected_positions)):
                    for j in range(i + 1, len(detected_positions)):
                        distance = np.linalg.norm(np.array(detected_positions[i]) - np.array(detected_positions[j]))
                        distances.append((distance, i, j))

                # Sort by distance and pick the closest pairs to determine the initial 3 dots
                distances.sort()
                selected_indices = set([distances[0][1], distances[0][2], distances[1][1], distances[1][2]])
                selected_indices = list(selected_indices)[:3]
                tracked_positions = [detected_positions[i] for i in selected_indices]

        else:
            # Subsequent frames: Only track the dots that maintain a consistent spatial relationship
            if len(detected_positions) >= 3:
                new_tracked_positions = []
                for pos in detected_positions:
                    # Check if this detected position could be one of the tracked dots based on distance
                    for tracked_pos in tracked_positions:
                        if np.linalg.norm(np.array(pos) - np.array(tracked_pos)) < MAX_DISTANCE_CHANGE:
                            new_tracked_positions.append(pos)
                            break
                if len(new_tracked_positions) == 3:
                    tracked_positions = new_tracked_positions

        if tracked_positions:
            # Calculate the centroid of the three tracked red dots
            centroid_x = int(np.mean([pos[0] for pos in tracked_positions]))
            centroid_y = int(np.mean([pos[1] for pos in tracked_positions]))
            centroid = (centroid_x, centroid_y)
            trajectory.append(centroid)
            cv2.circle(frame, centroid, 5, (255, 0, 0), -1)

        cv2.imshow('Tracking', frame)
        if output_video_path:
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if output_video_path:
        out.release()
    cv2.destroyAllWindows()

    # Optional: Smooth the trajectory if needed
    trajectory = smooth_trajectory(trajectory)
    
    return trajectory, width, height



def smooth_trajectory(trajectory, window_size=5):
    smoothed = []
    for i in range(len(trajectory)):
        if i < window_size:
            smoothed.append(trajectory[i])
        else:
            avg_x = np.mean([p[0] for p in trajectory[i-window_size:i]])
            avg_y = np.mean([p[1] for p in trajectory[i-window_size:i]])
            smoothed.append((int(avg_x), int(avg_y)))
    return smoothed
