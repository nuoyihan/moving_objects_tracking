import matplotlib.pyplot as plt
import numpy as np

# def visualize_trajectory(trajectory, video_path=None):
#     import matplotlib.pyplot as plt
    
#     plt.figure(figsize=(10, 6))
#     trajectory = np.array(trajectory)
#     plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='r', linestyle='-', linewidth=2, markersize=5)
#     plt.title('Red Dot Trajectory')
#     plt.xlabel('X Coordinate')
#     plt.ylabel('Y Coordinate')
#     plt.gca().invert_yaxis()  # Ensure this matches the video coordinate system
#     plt.grid(True)

#     if video_path:
#         output_image_path = video_path.replace('.MOV', '_trajectory.png')
#         plt.savefig(output_image_path)
#         print(f"Trajectory plot saved to {output_image_path}")
#     else:
#         plt.show()


# def visualize_trajectory(trajectory, video_path=None):
#     plt.figure(figsize=(10, 6))
    
#     # Convert trajectory list to a 2D NumPy array
#     trajectory = np.array(trajectory)

#     # Ensure the array has the correct shape
#     if len(trajectory) > 0 and trajectory.shape[1] == 2:
#         plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='r', linestyle='-', linewidth=2, markersize=5)
#     else:
#         print("Trajectory array does not have the correct shape")

#     plt.title('Red Dot Trajectory')
#     plt.xlabel('X Coordinate')
#     plt.ylabel('Y Coordinate')
#     plt.gca().invert_yaxis()
#     plt.grid(True)

#     if video_path:
#         output_image_path = video_path.replace('.MOV', '_trajectory.png')
#         plt.savefig(output_image_path)
#         print(f"Trajectory plot saved to {output_image_path}")
#     else:
#         plt.show()


def visualize_trajectory(trajectory, video_width, video_height, save_path):
    trajectory = np.array(trajectory)
    
    plt.figure(figsize=(video_width/100, video_height/100))  # Set the figure size proportional to the video size
    
    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='r', linestyle='-', linewidth=2, markersize=5)
    
    plt.gca().invert_yaxis()  # Invert y-axis so (0,0) is at the top-left
    plt.gca().set_aspect('equal', adjustable='box')  # Set the aspect ratio to be equal
    
    plt.xlim(0, video_width)
    plt.ylim(0, video_height)
    
    plt.title("Red Dot Trajectory")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)

    plt.savefig(save_path, bbox_inches='tight')
    plt.show()