from matplotlib import pyplot as plt
import cv2
import numpy as np

def detect_paper_get_crop_details(video_path, output_path=None):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Read the first frame from the video
    ret, frame = cap.read()

    if not ret:
        cap.release()
        raise ValueError("Failed to read the first frame from the video.")

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to get the white regions
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the largest rectangle is the white paper
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image
    cropped_image = frame[y:y+h, x:x+w]

    # # Display the cropped image without borders using OpenCV
    # cv2.imshow('Cropped Image', cropped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Display the cropped image without borders
    # fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    # ax.axis('off')

    # Create a figure without any borders
    fig, ax = plt.subplots(figsize=(w/ 100, h / 100), dpi=100)

    # Hide the figure frame and axes
    fig.patch.set_visible(False)
    ax.axis('off')
    
    # Set tight layout to avoid any padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Display the image
    ax.imshow(cropped_image)

    # Save the cropped image if an output path is provided
    if output_path:
        cv2.imwrite(output_path, cropped_image)

    # Prepare crop details
    crop_details = {
        "x_start": x,
        "y_start": y,
        "crop_width": w,
        "crop_height": h
    }

    # Release the video capture object
    cap.release()

    return cropped_image, crop_details

def crop_video_using_details(video_path, output_video_path, crop_details):
    """
    Crop the entire video using the provided crop details.

    Parameters:
    - video_path: str, path to the input video file
    - output_video_path: str, path to save the cropped video
    - crop_details: dict, containing the starting point (x, y), crop width, and crop height

    Returns:
    - None
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Crop parameters
    x_start = crop_details['x_start']
    y_start = crop_details['y_start']
    crop_width = crop_details['crop_width']
    crop_height = crop_details['crop_height']

    # Validate the crop parameters
    if x_start + crop_width > width or y_start + crop_height > height:
        cap.release()
        raise ValueError("Crop dimensions exceed the original video dimensions.")

    # Define the codec and create a VideoWriter object to save the cropped video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'MJPG', etc.
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (crop_width, crop_height))

    # Process each frame
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop the frame
        cropped_frame = frame[y_start:y_start+crop_height, x_start:x_start+crop_width]
        
        # Write the cropped frame to the output video
        out.write(cropped_frame)

    # Release everything when done
    cap.release()
    out.release()

    print(f"Cropped video saved to {output_video_path}")