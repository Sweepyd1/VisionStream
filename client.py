import numpy as np
import cv2 as cv
import socket
import struct
import time
import msgpack
import lz4.frame
from collections import deque

from RDP_algo import approxPolyDP  # твоя реализация RDP

# ---------------- СЕТЬ ----------------
host_ip = "192.168.2.148"
port = 9999

# ---------------- ЦЕЛЕВОЙ БИТРЕЙТ/КОНТРОЛЛЕР ----------------
TARGET_BITRATE_BPS = 500_000  # целевой ~0.2 Мбит/с
EMA_ALPHA = 0.25
HOLD_FRAMES = 20
UP_THRESH = int(TARGET_BITRATE_BPS * 1.20)
DOWN_THRESH = int(TARGET_BITRATE_BPS * 0.80)

# мягкое масштабирование порогов Canny
CANNY_MIN_SCALE = 0.70
CANNY_MAX_SCALE = 1.30

# коридор для epsilon
BASE_EPS = 0.3
OVER_EPS = 1.5

# фильтры «пылинок»
MIN_AREA = 50.0
MIN_ARCLEN = 40.0

# ---------------- ВИДЕО ----------------
target_width, target_height = 720, 480
video_source = "video.mp4"  # 0 для веб-камеры


def run_client():
    target_fps = 25
    frame_delay = 1.0 / target_fps

    # --- сетевое подключение ---
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    client_socket.connect((host_ip, port))

    # --- входное видео ---
    cap = cv.VideoCapture(video_source)
    cap.set(cv.CAP_PROP_FPS, target_fps)

    try:
        total_latency = 0.0
        frame_count = 0

        avg_bps = 0.0
        mode = "TREE"
        mode_hold = 0
        eps_smooth = BASE_EPS

        tx_window = deque()
        tx_bytes_sum = 0

        while True:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv.resize(
                frame, (target_width, target_height), interpolation=cv.INTER_AREA
            )

            # подавление зелёного HUD
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mask_hud = cv.inRange(hsv, (35, 50, 50), (85, 255, 255))
            frame[mask_hud > 0] = 0

            # grayscale + контраст + сглаживание
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            blurred = cv.GaussianBlur(gray, (3, 3), 0.8)

            # авто-пороги Canny
            v = np.median(blurred)
            low = int(max(0, (1.0 - 0.33) * v))
            high = int(min(255, (1.0 + 0.33) * v))

            # оценка загрузки
            load_ratio = (avg_bps - DOWN_THRESH) / max(1, (UP_THRESH - DOWN_THRESH))
            load_ratio = max(0.0, min(1.0, load_ratio))

            # масштабирование Canny
            scale = CANNY_MIN_SCALE + (CANNY_MAX_SCALE - CANNY_MIN_SCALE) * load_ratio
            low_c = int(max(0, low * scale))
            high_c = int(min(255, high * scale))

            edges = cv.Canny(blurred, low_c, high_c, apertureSize=3, L2gradient=True)

            # морфология
            kernel = np.ones((3, 3), np.uint8)
            binary = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=2)

            # поиск контуров
            retr_mode = cv.RETR_TREE
            contours, hierarchy = cv.findContours(
                binary, retr_mode, cv.CHAIN_APPROX_SIMPLE
            )

            # плавный epsilon
            eps_target = BASE_EPS + (OVER_EPS - BASE_EPS) * load_ratio
            eps_smooth = 0.9 * eps_smooth + 0.1 * eps_target
            eps = eps_smooth

            # фильтрация и упрощение
            filtered_contours = []
            for cnt in contours:
                pts = cnt.reshape(-1, 2).astype(np.float32)

                # прореживание длинных контуров
                if len(pts) > 100:
                    step = max(1, len(pts) // 100)
                    pts = pts[::step]

                if pts.shape[0] < 3:
                    continue
                if cv.contourArea(pts.astype(np.int32)) < MIN_AREA:
                    continue
                if cv.arcLength(pts.astype(np.int32), True) < MIN_ARCLEN:
                    continue

                approx = approxPolyDP(pts, epsilon=eps, closed=True)
                if approx.shape[0] >= 3:
                    filtered_contours.append(approx.astype(np.int16))

            # сериализация и сжатие
            serializable_contours = [cnt.tolist() for cnt in filtered_contours]
            serialized = msgpack.packb(serializable_contours, use_bin_type=True)
            compressed = lz4.frame.compress(serialized)
            message_size = struct.pack("Q", len(compressed))
            bytes_this_frame = len(message_size) + len(compressed)

            client_socket.sendall(message_size + compressed)

            # обновляем окно реального битрейта
            now = time.time()
            tx_window.append((now, bytes_this_frame))
            tx_bytes_sum += bytes_this_frame
            while tx_window and (now - tx_window[0][0]) > 1.0:
                _, old_b = tx_window.popleft()
                tx_bytes_sum -= old_b
            window_secs = max(1e-3, (tx_window[-1][0] - tx_window[0][0]))
            real_bps = (tx_bytes_sum * 8) / window_secs

            avg_bps = (1.0 - EMA_ALPHA) * avg_bps + EMA_ALPHA * real_bps

            # управление FPS по битрейту
            if real_bps > UP_THRESH:
                target_fps = max(10, target_fps - 2)
            elif real_bps < DOWN_THRESH:
                target_fps = min(25, target_fps + 1)
            frame_delay = 1.0 / target_fps

            # задержка
            processing_time = now - frame_start
            sleep_time = frame_delay - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)

            # статистика
            frame_count += 1
            total_latency += processing_time
            avg_latency = total_latency / frame_count
            bitrate_mbps = real_bps / 1_000_000

            if frame_count % 15 == 0:
                print(
                    f"eps={eps:.2f}  avg_bps={int(avg_bps)}  real_bps={int(real_bps)} "
                    f"FPS={target_fps} Контуров={len(filtered_contours)}"
                )

            print(
                f"[Битрейт] Реальный: {real_bps / 1000:.1f} кбит/с | "
                f"Средний: {avg_bps / 1000:.1f} кбит/с"
            )
            # предпросмотр
            contour_img = np.zeros((target_height, target_width), dtype=np.uint8)
            for cnt in filtered_contours:
                try:
                    reshaped = cnt.reshape(-1, 1, 2).astype(np.int32)
                    if reshaped.shape[0] > 0:
                        cv.drawContours(contour_img, [reshaped], -1, 255, 1)
                except Exception as e:
                    print(f"Ошибка при отрисовке: {e}")

            # cv.imshow("Preview", contour_img)
            # if cv.waitKey(1) & 0xFF == ord('q'):
            #     break

    except (KeyboardInterrupt, socket.error):
        print("Отключение клиента...")
    finally:
        cap.release()
        cv.destroyAllWindows()
        client_socket.close()


if __name__ == "__main__":
    run_client()

# cd D:\HSE\Subjects\C\contours_video
# python server.py
# python client.py
