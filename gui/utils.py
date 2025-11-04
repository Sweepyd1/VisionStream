import threading
import queue

params = {
    "SEND_EVERY_N_FRAMES": 5,
    "MAX_PRIMITIVES": 100,
    "EPSILON_FACTOR": 0.007,
    "MIN_CONTOUR_AREA": 350,
    "VIDEO_PATH": "../video3.mp4",
    "HOST_IP": "127.0.0.1",
    "PORT": 9999,
    "INPUT_WIDTH": 640,
    "INPUT_HEIGHT": 360,
    "DISPLAY_WIDTH": 1920,
    "DISPLAY_HEIGHT": 1080,
    "HISTORY_FRAMES": 6,
    "DECAY_FACTOR": 0.8,
}

server_control = threading.Event()
client_control = threading.Event()
client_bitrate_log = []  
log_lock = threading.Lock()
server_frame_queue = queue.Queue(maxsize=2)

stats = {
    "bitrate_mbps": 0.0,
    "fps": 0.0,
    "primitive_count": 0,
    "last_update": 0.0,
}
stats_lock = threading.Lock()


stats_buffers = {
    "frame_times": [],
    "bytes_history": [],
}
stats_buffers_lock = threading.Lock()