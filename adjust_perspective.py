import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def filter_and_cluster_lines(lines, threshold=20, angle_threshold=10):
    """
    Filter out lines that are too close to each other and cluster them based on orientation.

    Parameters:
    - lines: List of lines detected by Hough Transform.
    - threshold: Distance threshold below which lines are considered duplicates.
    - angle_threshold: Angle threshold in degrees to consider lines as similar in orientation.

    Returns:
    - clustered_lines: List of clustered and filtered lines.
    """
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if -angle_threshold < angle < angle_threshold:  # Horizontal lines
                horizontal_lines.append(line)
            elif 90 - angle_threshold < angle < 90 + angle_threshold or \
                 -90 - angle_threshold < angle < -90 + angle_threshold:  # Vertical lines
                vertical_lines.append(line)

    # Function to filter lines based on distance
    def filter_lines(lines):
        filtered = []
        for line in lines:
            for filtered_line in filtered:
                if np.linalg.norm(line[0] - filtered_line[0]) < threshold:
                    break
            else:
                filtered.append(line)
        return filtered

    filtered_horizontal = filter_lines(horizontal_lines)
    filtered_vertical = filter_lines(vertical_lines)

    return filtered_horizontal + filtered_vertical

def enhance_grid_detection_with_strong_filtering(video_path, output_video_path):
    """
    Enhance the grid detection by increasing the contrast of the video, filtering redundant lines strongly,
    and adjusting the perspective of the video. Save the adjusted video.

    Parameters:
    - video_path: str, path to the input video file
    - output_video_path: str, path to save the perspective-adjusted video

    Returns:
    - None
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Read the first frame from the video
    ret, frame = cap.read()

    if not ret:
        cap.release()
        raise ValueError("Failed to read the first frame from the video.")

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Increase contrast using histogram equalization
    contrast_enhanced = cv2.equalizeHist(gray)

    # Use GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(contrast_enhanced, (5, 5), 0)

    # Use Canny edge detection to find edges
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=50)

    if lines is None:
        cap.release()
        raise ValueError("Failed to detect grid lines in the first frame.")

    # Filter and cluster lines
    filtered_lines = filter_and_cluster_lines(lines, threshold=20, angle_threshold=10)

    # Draw the filtered lines on the original frame for visualization
    debug_frame = frame.copy()
    for line in filtered_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Find intersections of the filtered lines
    def line_intersections(lines):
        points = []
        for line1 in lines:
            for line2 in lines:
                x1, y1, x2, y2 = line1[0]
                x3, y3, x4, y4 = line2[0]
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if denom != 0:
                    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
                    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
                    points.append([px, py])
        return points

    intersections = line_intersections(filtered_lines)

    # Filter the intersections to find the corners of the grid
    intersections = np.array(intersections)
    if len(intersections) < 4:
        cap.release()
        raise ValueError("Failed to detect enough grid intersections for perspective adjustment.")
    
    # Sort intersections to get the four corners of the grid
    s = intersections.sum(axis=1)
    tl = intersections[np.argmin(s)]
    br = intersections[np.argmax(s)]
    diff = np.diff(intersections, axis=1)
    tr = intersections[np.argmin(diff)]
    bl = intersections[np.argmax(diff)]

    corners_detected = np.float32([tl, tr, bl, br])

    # Draw the detected corners on the debug frame
    for point in corners_detected:
        cv2.circle(debug_frame, tuple(int(v) for v in point), 10, (255, 0, 0), -1)

    # Display the debug frame with detected corners
    cv2.imshow('Detected Corners', debug_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Define the desired positions for the corners (these should form an upright rectangle)
    corners_upright = np.float32([[0, 0], [frame.shape[1] - 1, 0], [0, frame.shape[0] - 1], [frame.shape[1] - 1, frame.shape[0] - 1]])

    # Get the perspective transformation matrix
    M = cv2.getPerspectiveTransform(corners_detected, corners_upright)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create a VideoWriter object to save the adjusted video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Apply the perspective transformation to each frame
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Increase contrast for each frame before transforming
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contrast_enhanced_frame = cv2.equalizeHist(gray_frame)

        # Apply the perspective transformation
        adjusted_frame = cv2.warpPerspective(contrast_enhanced_frame, M, (width, height))
        
        # Convert back to BGR (3-channel image) for video writing
        adjusted_frame_bgr = cv2.cvtColor(adjusted_frame, cv2.COLOR_GRAY2BGR)
        
        # Write the adjusted frame to the output video
        out.write(adjusted_frame_bgr)

    # Release everything when done
    cap.release()
    out.release()

    print(f"Perspective-adjusted video saved to {output_video_path}")



def manual_perspective_adjustment(video_path, output_video_path, corners_detected):
    """
    Manually adjust the perspective of the video using predefined corners.
    Saves the adjusted video.

    Parameters:
    - video_path: str, path to the input video file
    - output_video_path: str, path to save the perspective-adjusted video
    - corners_detected: list, list of four corner points (tl, tr, bl, br)

    Returns:
    - None
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create a VideoWriter object to save the adjusted video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Define the desired positions for the corners (these should form an upright rectangle)
    corners_upright = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])

    # Get the perspective transformation matrix
    M = cv2.getPerspectiveTransform(np.float32(corners_detected), corners_upright)

    # Apply the perspective transformation to each frame
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Apply the perspective transformation
        adjusted_frame = cv2.warpPerspective(frame, M, (width, height))
        
        # Write the adjusted frame to the output video
        out.write(adjusted_frame)

    # Release everything when done
    cap.release()
    out.release()

    print(f"Perspective-adjusted video saved to {output_video_path}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="input video path")
    parser.add_argument("output_path", type=str, help="output video path")
    args = parser.parse_args()

# enhance_grid_detection_with_strong_filtering(args.input_path, args.output_path)
manual_perspective_adjustment()