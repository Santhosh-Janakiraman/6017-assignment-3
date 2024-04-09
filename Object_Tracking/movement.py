import csv
import imutils
import cv2
import json
import math
import numpy as np
from config import VIDEO_CONFIG
from itertools import zip_longest
from math import ceil
from scipy.spatial.distance import euclidean
from colors import RGB_COLORS, gradient_color_RGB

tracks = []
with open('processed_data/movement_data.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        if len(row[3:]) > 4:
            temp = []
            data = row[3:]
            for i in range(0, len(data), 2):
                temp.append([int(data[i]), int(data[i+1])])
            tracks.append(temp)

with open('processed_data/video_data.json', 'r') as file:
	data = json.load(file)
	vid_fps = data["VID_FPS"]
	data_record_frame = data["DATA_RECORD_FRAME"]
	frame_size = data["PROCESSED_FRAME_SIZE"]

cap = cv2.VideoCapture(VIDEO_CONFIG["VIDEO_CAP"])
cap.set(1, 100)
(ret, tracks_frame) = cap.read()
tracks_frame = imutils.resize(tracks_frame, width=frame_size)
heatmap_frame = np.copy(tracks_frame)
print(tracks_frame.shape)
stationary_threshold_seconds = 2
stationary_threshold_frame =  round(vid_fps * stationary_threshold_seconds / data_record_frame)
stationary_distance = frame_size * 0.05
max_stationary_time = 120
blob_layer = 50
max_blob_size = frame_size * 0.1
layer_size = max_blob_size / blob_layer
color_start = 210
color_end = 0
color_steps = int((color_start - color_end) / blob_layer)
scale = 1.5

# print(stationary_distance)
# print(stationary_threshold_frame)

stationary_points = []
movement_points = []
total = 0
for movement in tracks:
    temp_movement_point = [movement[0]]
    stationary = movement[0]
    stationary_time = 0
    for i in movement[1:]:
        if euclidean(stationary, i) < stationary_distance:
            stationary_time += 1
        else:
            temp_movement_point.append(i)
            if stationary_time > stationary_threshold_frame:
                stationary_points.append([stationary, stationary_time])
            stationary = i
            stationary_time = 0
    movement_points.append(temp_movement_point)
    total += len(temp_movement_point)

# print(total)
# print(movement_points)

color1 = (255, 96, 0)
color2 = (0, 28, 255)
for track in movement_points:
    for i in range(len(track) - 1):
        color = gradient_color_RGB(color1, color2, len(track) - 1, i)
        cv2.line(tracks_frame, tuple(track[i]), tuple(track[i+1]), color, 2)
    
def draw_blob(frame, coordinates, time):
    if time >= max_stationary_time:
        layer = blob_layer
    else:
        layer = math.ceil(time * scale / layer_size)
    for x in reversed(range(layer)):
        color = color_start - (color_steps * x)
        size = x * layer_size
        cv2.circle(frame, coordinates, int(size), (color, color, color), -1)

heatmap = np.zeros((heatmap_frame.shape[0], heatmap_frame.shape[1]), dtype=np.uint8)
for points in stationary_points:
    draw_heatmap = np.zeros((heatmap_frame.shape[0], heatmap_frame.shape[1]), dtype=np.uint8)
    draw_blob(draw_heatmap, tuple(points[0]), points[1])
    heatmap = cv2.add(heatmap, draw_heatmap)

lo = np.array([color_start])
hi = np.array([255])
mask = cv2.inRange(heatmap, lo, hi)
heatmap[mask > 0] = color_start

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
lo = np.array([128,0,0])
hi = np.array([136,0,0])
mask = cv2.inRange(heatmap, lo, hi)
heatmap[mask > 0] = (0, 0, 0)

for row in range(heatmap.shape[0]):
    for col in range(heatmap.shape[1]):
        if (heatmap[row][col] == np.array([0,0,0])).all():
            heatmap[row][col] = heatmap_frame[row][col] 

heatmap_frame = cv2.addWeighted(heatmap, 0.75, heatmap_frame, 0.25, 1)





# # ------- frame with gaps and header

# header_height = 50

# # Create black headers for each frame
# header_tracks = np.zeros((header_height, tracks_frame.shape[1], 3), dtype=np.uint8)
# header_heatmap = np.zeros((header_height, heatmap_frame.shape[1], 3), dtype=np.uint8)

# # Add text to each header
# cv2.putText(header_tracks, 'Movement Tracks', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
# cv2.putText(header_heatmap, 'Stationary Location Heatmap', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# # Attach the headers to the top of each frame
# tracks_frame_with_header = np.vstack((header_tracks, tracks_frame))
# heatmap_frame_with_header = np.vstack((header_heatmap, heatmap_frame))

# # Create a separator (black bar) to add between the frames
# separator_height = 10
# separator = np.zeros((separator_height, tracks_frame.shape[1], 3), dtype=np.uint8)

# # Stack frames with separator
# combined_frame = np.vstack((tracks_frame_with_header, separator, heatmap_frame_with_header))


# # combined_frame = np.vstack((tracks_frame, heatmap_frame))


# # Show the combined image in a single window
# cv2.imshow("Engagement Tracking", combined_frame)

# # Loop to keep the window open until 'ESC' key is pressed
# while True:
#     # Wait for the ESC key to be pressed
#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# # Close all OpenCV windows and release the video capture object
# cv2.destroyAllWindows()
# cap.release()  



# --------------------------

header_height = 50
separator_height = 10

# Convert the original frame to grayscale (for the greyed-out effect)
greyed_out_frame = cv2.cvtColor(tracks_frame, cv2.COLOR_BGR2GRAY)
greyed_out_frame_colored = cv2.cvtColor(greyed_out_frame, cv2.COLOR_GRAY2BGR)

# Process for stationary points on heatmap
for points in stationary_points:
    draw_blob(heatmap, tuple(points[0]), points[1])

# Normalize the heatmap
normalized_heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
colored_heatmap = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)

# Blend the heatmap with the greyed-out frame
alpha = 0.5  # Adjust alpha for blending
for row in range(greyed_out_frame_colored.shape[0]):
    for col in range(greyed_out_frame_colored.shape[1]):
        if np.any(colored_heatmap[row][col] != [0, 0, 0]):
            # greyed_out_frame_colored[row][col] = greyed_out_frame_colored[row][col] * (1 - alpha) 
            # + colored_heatmap[row][col] * alpha
             greyed_out_frame_colored[row][col] = colored_heatmap[row][col] * alpha

# Enhance stationary points
bright_red = (0, 0, 255)  # Bright red color
for point, time in stationary_points:
    cv2.circle(greyed_out_frame_colored, tuple(point), 10, bright_red, -1)  # Fixed size for visibility

# Process for movement tracking (on a copy of the original frame)
tracks_frame_movement = tracks_frame.copy()
for track in movement_points:
    for i in range(len(track) - 1):
        color = gradient_color_RGB(color1, color2, len(track) - 1, i)
        cv2.line(tracks_frame_movement, tuple(track[i]), tuple(track[i+1]), color, 2)

# Combine heatmap and movement tracking frames horizontally
        

# Create black headers for each frame
header_tracks = np.zeros((header_height, tracks_frame_movement.shape[1], 3), dtype=np.uint8)
header_heatmap = np.zeros((header_height, greyed_out_frame_colored.shape[1], 3), dtype=np.uint8)

# Add text to each header
cv2.putText(header_tracks, 'Movement Tracks', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(header_heatmap, 'Stationary Location Heatmap', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Attach the headers to the top of each frame
tracks_frame_with_header = np.vstack((header_tracks, tracks_frame_movement))
heatmap_frame_with_header = np.vstack((header_heatmap, greyed_out_frame_colored))

# Create a separator (black bar) to add between the frames
separator = np.zeros((separator_height, tracks_frame_movement.shape[1], 3), dtype=np.uint8)

# Stack frames with separator
combined_frame = np.vstack((tracks_frame_with_header, separator, heatmap_frame_with_header))


# combined_frame = np.vstack((tracks_frame_movement, greyed_out_frame_colored))

# Display the combined frame
cv2.imshow("Combined Movement and Heatmap", combined_frame)

# Loop to keep the window open until 'ESC' key is pressed
while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
cap.release()


#--------------------------
#--------------------------

