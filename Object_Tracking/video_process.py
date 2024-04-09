
import time
import datetime
import numpy as np
import imutils
import cv2
import time
from math import ceil
from scipy.spatial.distance import euclidean
from tracking import detect_human
from util import rect_distance, progress, kinetic_energy
from colors import RGB_COLORS
from config import SHOW_DETECT, DATA_RECORD, SHOW_TRACKING_ID
from config import	SHOW_PROCESSING_OUTPUT, YOLO_CONFIG, VIDEO_CONFIG, DATA_RECORD_RATE


from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

from deep_sort import generate_detections as gdet
IS_CAM = VIDEO_CONFIG["IS_CAM"]
HIGH_CAM = VIDEO_CONFIG["HIGH_CAM"]

def _record_movement_data(movement_data_writer, movement):
	track_id = movement.track_id 
	entry_time = movement.entry 
	exit_time = movement.exit			
	positions = movement.positions
	positions = np.array(positions).flatten()
	positions = list(positions)
	data = [track_id] + [entry_time] + [exit_time] + positions
	movement_data_writer.writerow(data)

def _record_attendee_data(time, human_count, attendee_data_writer):
	data = [time, human_count]
	attendee_data_writer.writerow(data)
 
def _end_video(tracker, frame_count, movement_data_writer):
	for t in tracker.tracks:
		if t.is_confirmed():
			t.exit = frame_count
			_record_movement_data(movement_data_writer, t)
		

def video_process(cap, frame_size, net, ln, encoder, tracker, movement_data_writer, attendee_data_writer):
	def _calculate_FPS():
		t1 = time.time() - t0
		VID_FPS = frame_count / t1

	if IS_CAM:
		VID_FPS = None
		DATA_RECORD_FRAME = 1
		TIME_STEP = 1
		t0 = time.time()
	else:
		VID_FPS = cap.get(cv2.CAP_PROP_FPS)
		DATA_RECORD_FRAME = int(VID_FPS / DATA_RECORD_RATE)
		TIME_STEP = DATA_RECORD_FRAME/VID_FPS

	frame_count = 0
	display_frame_count = 0

	RE = False

	while True:
		(ret, frame) = cap.read()

		# Stop the loop when video ends
		if not ret:
			_end_video(tracker, frame_count, movement_data_writer)
			if not VID_FPS:
				_calculate_FPS()
			break

		# Update frame count
		if frame_count > 1000000:
			if not VID_FPS:
				_calculate_FPS()
			frame_count = 0
			display_frame_count = 0
		frame_count += 1
		
		# Skip frames according to given rate
		if frame_count % DATA_RECORD_FRAME != 0:
			continue

		display_frame_count += 1

		# Resize Frame to given size
		frame = imutils.resize(frame, width=frame_size)

		# Get current time
		current_datetime = datetime.datetime.now()

		# Run detection algorithm
		if IS_CAM:
			record_time = current_datetime
		else:
			record_time = frame_count
		
		# Run tracking algorithm
		[humans_detected, expired] = detect_human(net, ln, frame, encoder, tracker, record_time)

		# Record movement data
		for movement in expired:
			_record_movement_data(movement_data_writer, movement)
		
		
			
		# Initiate video process loop
		if SHOW_PROCESSING_OUTPUT or SHOW_DETECT:

			# Initialize set for violate so an individual will be recorded only once
			
			violate_set = set()
			# Initialize list to record violation count for each individual detected
			violate_count = np.zeros(len(humans_detected))

			# Initialize list to record id of individual with abnormal energy level
			abnormal_individual = []
			ABNORMAL = False
			for i, track in enumerate(humans_detected):
				# Get object bounding box
				[x, y, w, h] =  list(map(int, track.to_tlbr().tolist()))
				# Get object centroid
				[cx, cy] = list(map(int, track.positions[-1]))
				# Get object id
				idx = track.track_id
	
				
				if SHOW_DETECT and not RE:
					# cv2.rectangle(frame, (x, y), (w, h), RGB_COLORS["green"], 2)
					cv2.rectangle(frame, (x, y), (w, h), RGB_COLORS["red"], 2)
				

				if SHOW_TRACKING_ID:
					cv2.putText(frame, str(int(idx)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RGB_COLORS["green"], 2)
			
		# Display crowd count on screen
		if SHOW_DETECT:
			text = "Crowd count: {}".format(len(humans_detected))
			cv2.putText(frame, text, (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
	
		# Record crowd data to file
		if DATA_RECORD:
			_record_attendee_data(record_time, len(humans_detected), attendee_data_writer)

		# Display video output or processing indicator
		if SHOW_PROCESSING_OUTPUT:
			cv2.imshow("Processed Output", frame)
		else:
			progress(display_frame_count)

		# Press 'Q' to stop the video display
		if cv2.waitKey(1) & 0xFF == ord('q'):
			# Record the movement when video ends
			_end_video(tracker, frame_count, movement_data_writer)
			# Compute the processing speed
			if not VID_FPS:
				_calculate_FPS()
			break
	
	cv2.destroyAllWindows()
	return VID_FPS
