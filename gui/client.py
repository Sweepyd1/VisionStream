import numpy as np
import cv2 as cv
import socket
import struct
import time
import msgpack
import zlib
import os
import sys


sys.path.append(os.path.dirname(__file__))
from utils import params, client_control, log_lock

def connect_with_retry():
    host_ip = params["HOST_IP"]
    port = params["PORT"]
    while client_control.is_set():
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.connect((host_ip, port))
            with log_lock:
                print("✅ Подключено к серверу")
            return sock
        except Exception as e:
            with log_lock:
                print(f"⏳ Ожидание сервера... ({e})")
            time.sleep(1)
    return None

def simplify_and_classify(frame):
    TARGET_RES = (params["INPUT_WIDTH"], params["INPUT_HEIGHT"])
    MAX_PRIMITIVES = params["MAX_PRIMITIVES"]
    EPSILON_FACTOR = params["EPSILON_FACTOR"]
    MIN_CONTOUR_AREA = params["MIN_CONTOUR_AREA"]

    small = cv.resize(frame, TARGET_RES, interpolation=cv.INTER_AREA)
    gray = cv.cvtColor(small, cv.COLOR_BGR2GRAY)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv.GaussianBlur(enhanced, (3, 3), 0)
    edges = cv.Canny(blurred, threshold1=25, threshold2=65)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    primitives = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        epsilon = EPSILON_FACTOR * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            angles = []
            for i in range(4):
                a = pts[i]
                b = pts[(i + 1) % 4]
                c = pts[(i + 2) % 4]
                ba = a - b
                bc = c - b
                denom = np.linalg.norm(ba) * np.linalg.norm(bc)
                if denom < 1e-6:
                    continue
                cosine_angle = np.dot(ba, bc) / denom
                cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
                angle = np.arccos(cosine_angle)
                angles.append(angle)
            if len(angles) == 4 and all(1.0 < a < 2.2 for a in angles):
                primitives.append(('rect', approx.reshape(-1, 2).tolist()))
                continue

        if len(approx) >= 3:
            primitives.append(('contour', approx.reshape(-1, 2).tolist()))


    def get_area(item):
        _, pts = item
        return cv.contourArea(np.array(pts, dtype=np.float32))

    primitives.sort(key=get_area, reverse=True)
    return primitives[:MAX_PRIMITIVES]

def run_client_gui():
    from time import time

    SEND_EVERY_N_FRAMES = params["SEND_EVERY_N_FRAMES"]
    VIDEO_PATH = params["VIDEO_PATH"]

    if not os.path.exists(VIDEO_PATH):
        with log_lock:
            print(f"❌ Видеофайл не найден: {VIDEO_PATH}")
        return

    cap = cv.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        with log_lock:
            print("❌ Не удалось открыть видеофайл")
        return

    client_socket = connect_with_retry()
    if client_socket is None:
        cap.release()
        return

    frame_count = 0
    bytes_sent_total = 0
    start_time = time()

    try:
        while client_control.is_set():
            ret, frame = cap.read()
            if not ret:
                with log_lock:
                    print("🎥 Видео закончилось. Клиент останавливается.")
                break

            if frame_count % SEND_EVERY_N_FRAMES != 0:
                frame_count += 1
                continue

            primitives = simplify_and_classify(frame)

            try:
                serialized = msgpack.packb(primitives, use_bin_type=True)
                compressed = zlib.compress(serialized, level=9)
                message = struct.pack("Q", len(compressed)) + compressed
                client_socket.sendall(message)
                bytes_sent_total += len(message)
            except Exception as e:
                with log_lock:
                    print(f"❌ Ошибка отправки данных: {e}")
                break

            elapsed = time() - start_time
            if elapsed > 1.0:
                bitrate_mbps = (bytes_sent_total * 8) / (elapsed * 1_000_000)
                log_msg = f"📡 Битрейт: {bitrate_mbps:.3f} Мбит/с | Примитивов: {len(primitives)}"
                with log_lock:
                    print(log_msg)
                bytes_sent_total = 0
                start_time = time()

            frame_count += 1

    except Exception as e:
        with log_lock:
            print(f"🛑 Клиент прерван: {e}")
    finally:
        cap.release()
        try:
            client_socket.close()
        except:
            pass
        with log_lock:
            print("⏹ Клиент остановлен")