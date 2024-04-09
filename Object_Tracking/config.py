import datetime

VIDEO_CONFIG = {
	"VIDEO_CAP" : "Object_Tracking/video/basketball_game.mp4",
	"IS_CAM" : False,
	"CAM_APPROX_FPS": 2,
	"HIGH_CAM": False,
	"START_TIME": datetime.datetime.now()
}

# Load YOLOv3-tiny weights and config
YOLO_CONFIG = {
	"WEIGHTS_PATH" : "Object_Tracking/model_data/model/model_trained.weights",
	"CONFIG_PATH" : "Object_Tracking/model_data/model/model_trained.cfg"
}


# Show individuals detected
SHOW_PROCESSING_OUTPUT = True
# Show individuals detected
SHOW_DETECT = True
# Data record
DATA_RECORD = True
# Data record rate (data record per frame)
DATA_RECORD_RATE = 5
# Show tracking id
SHOW_TRACKING_ID = False


MIN_CONF = 0.3
# Threshold for Non-maxima surpression
NMS_THRESH = 0.2
# Resize frame for processing
FRAME_SIZE = 1080
# Tracker max missing age before removing (seconds)
TRACK_MAX_AGE = 3