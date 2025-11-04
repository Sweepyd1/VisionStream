import socket
import time
import cv2 as cv
import zlib
import struct
import msgpack
import numpy as np
import sys
import os
from utils import params, server_control, server_frame_queue, log_lock, stats_lock, stats, stats_buffers, stats_buffers_lock

sys.path.append(os.path.dirname(__file__))
port = 9999
INPUT_WIDTH, INPUT_HEIGHT = 640, 360
DISPLAY_WIDTH, DISPLAY_HEIGHT = 1920, 1080

HISTORY_FRAMES = 6
DECAY_FACTOR = 0.8

def run_server_gui():
    global server_frame_queue
    import queue
   

    port = params["PORT"]
    INPUT_WIDTH, INPUT_HEIGHT = params["INPUT_WIDTH"], params["INPUT_HEIGHT"]
    DISPLAY_WIDTH, DISPLAY_HEIGHT = params["DISPLAY_WIDTH"], params["DISPLAY_HEIGHT"]
    HISTORY_FRAMES = params["HISTORY_FRAMES"]
    DECAY_FACTOR = params["DECAY_FACTOR"]

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", port))
    server_socket.listen(1)
    print(f"📡 Сервер запущен на порту {port}")
    with stats_buffers_lock:
        stats_buffers["frame_times"] = []
        stats_buffers["bytes_history"] = []

    contour_history = []

    try:
        client_socket, addr = server_socket.accept()
        client_socket.settimeout(5.0)
        print(f"🔌 Подключён: {addr}")

        data = b""
        payload_size = struct.calcsize("Q")

        while server_control.is_set():
            try:
                while len(data) < payload_size:
                    packet = client_socket.recv(4096)
                    if not packet:
                        raise ConnectionError("Пустой пакет")
                    data += packet

                msg_size = struct.unpack("Q", data[:payload_size])[0]
                data = data[payload_size:]

                while len(data) < msg_size:
                    packet = client_socket.recv(4096)
                    if not packet:
                        raise ConnectionError("Разрыв")
                    data += packet

                compressed = data[:msg_size]
                data = data[msg_size:]

                decompressed = zlib.decompress(compressed)
                primitives = msgpack.unpackb(decompressed, raw=False)

                with log_lock:
                    print(f"✅ Получено примитивов: {len(primitives)}")

                contour_history.append(primitives)
                if len(contour_history) > HISTORY_FRAMES:
                    contour_history.pop(0)

                canvas = np.zeros((INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.uint8)

                for i, frame_primitives in enumerate(contour_history):
                    alpha = DECAY_FACTOR ** (len(contour_history) - 1 - i)
                    intensity = int(255 * alpha)
                    for ptype, pts in frame_primitives:
                        pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                        if pts_np.size == 0:
                            continue
                        pts_np = np.clip(pts_np, [0, 0], [INPUT_WIDTH - 1, INPUT_HEIGHT - 1])
                        if ptype == 'rect':
                            color = (255, 255, 255)
                            thickness = max(1, int(2 * alpha))
                        else:
                            color = (255, 255, 255)
                            thickness = max(1, int(1.5 * alpha))
                        cv.polylines(canvas, [pts_np], isClosed=True, color=color, thickness=thickness)
                
                current_time = time.time()

                with stats_buffers_lock:
                   
                    stats_buffers["frame_times"].append(current_time)
                    stats_buffers["frame_times"] = [t for t in stats_buffers["frame_times"] if current_time - t <= 1.0]

                    msg_size = len(compressed)
                    stats_buffers["bytes_history"].append((current_time, msg_size))
                    stats_buffers["bytes_history"] = [
                        (t, sz) for t, sz in stats_buffers["bytes_history"] if current_time - t <= 1.0
                    ]

                    ft = stats_buffers["frame_times"]
                    bh = stats_buffers["bytes_history"]

                    fps = (len(ft) - 1) / (ft[-1] - ft[0] + 1e-6) if len(ft) >= 2 else 0.0
                    total_bytes = sum(sz for _, sz in bh)
                    time_span = bh[-1][0] - bh[0][0] if len(bh) >= 2 else 1e-6
                    bitrate_mbps = (total_bytes * 8) / (time_span * 1_000_000 + 1e-6) if len(bh) >= 2 else 0.0

                with stats_lock:
                    stats["bitrate_mbps"] = bitrate_mbps
                    stats["fps"] = fps
                    stats["primitive_count"] = len(primitives)
                
                canvas_rgb = cv.cvtColor(canvas, cv.COLOR_BGR2RGB)
                canvas_display = cv.resize(canvas_rgb, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv.INTER_LINEAR)
                if server_frame_queue.full():
                    try:
                        server_frame_queue.get_nowait()
                    except Exception:
                        pass
                server_frame_queue.put(canvas_display)

                client_socket.sendall(b"ACK")

            except Exception as e:
                with log_lock:
                    print(f"⚠️ Ошибка сервера: {e}")
                break

    except Exception as e:
        with log_lock:
            print(f"🛑 Сервер завершён с ошибкой: {e}")
    finally:
        try:
            client_socket.close()
        except Exception:
            pass
        server_socket.close()