import numpy as np
import cv2 as cv
import socket
import struct
import time
import pickle
import zlib
from collections import deque

from RDP_algo import approxPolyDP  # твоя реализация RDP

# ---------------- СЕТЬ ----------------
host_ip = "192.168.0.102"
port = 9999

# ---------------- КОНФИГ КАНАЛА ----------------
TARGET_BITRATE_BPS = 300_000     # целевой битрейт (навигация: повыше деталей)
PEAK_FACTOR       = 2.0          # терпим пики до 2× целевого (~0.6 Мбит/с)

# Пороги гистерезиса по сглаженному битрейту (см. ниже)
UP_THRESH   = int(TARGET_BITRATE_BPS * 1.20)  # переключиться на EXT, если устойчиво выше
DOWN_THRESH = int(TARGET_BITRATE_BPS * 0.80)  # вернуться на TREE, если устойчиво ниже

# EMA-сглаживание реального битрейта (0..1): больше — быстрее реагируем, но дергаемся
EMA_ALPHA = 0.25

# RDP: диапазон упрощения (меньше eps — больше точек/деталей)
BASE_EPS = 0.5
OVER_EPS = 1.2

# Отсев «пылинок» до RDP (чтоб не гонять крошки)
MIN_AREA   = 5.0    # минимальная площадь контура (px^2)
MIN_ARCLEN = 10.0   # минимальный периметр (px)

# ---------------- ВИДЕО ----------------
target_width, target_height = 720, 480   # согласуй с канвой на сервере (canvas 480x720)
video_source = 0           # поставь 0 для веб-камеры

def run_client():
    bytes_sent  = 0
    target_fps  = 25
    frame_delay = 1.0 / target_fps

    # --- сетевое подключение ---
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    client_socket.connect((host_ip, port))

    # --- входное видео ---
    cap = cv.VideoCapture(video_source)
    cap.set(cv.CAP_PROP_FPS, target_fps)  # для файла часто не влияет — ниже делаем ручной sleep

    try:
        total_latency = 0.0   # накопленная задержка всех кадров
        frame_count   = 0     # сколько кадров обработали

        # Состояние контроллера качества
        avg_bps    = 0.0      # сглаженный по EMA «реальный битрейт»
        mode       = 'TREE'   # текущий режим контуров: TREE (все) или EXT (только внешние)
        mode_hold  = 0        # таймер удержания режима (анти-дрыг)
        eps_smooth = BASE_EPS # сглаженный epsilon для RDP (плавное упрощение)

        # «Окно последней секунды» для расчёта реального битрейта
        tx_window    = deque()  # будем хранить (timestamp, bytes_this_frame)
        tx_bytes_sum = 0        # сумма байт внутри окна

        while True:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # --- унификация размера кадра ---
            # Без этого координаты контуров могут «вылезать» за канву сервера.
            frame = cv.resize(frame, (target_width, target_height), interpolation=cv.INTER_AREA)

            # --- (опционально) подавляем зелёный HUD, чтобы не лез в контуры ---
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mask_hud = cv.inRange(hsv, (35, 50, 50), (85, 255, 255))
            frame[mask_hud > 0] = 0

            # --- gray + локальный контраст (CLAHE) + лёгкое сглаживание ---
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # 
            gray = clahe.apply(gray)
            blurred = cv.GaussianBlur(gray, (3, 3), 0.8)

            # --- Canny с авто-порогами от медианы ---
            v = np.median(blurred)
            low  = int(max(0,   (1.0 - 0.33) * v))
            high = int(min(255, (1.0 + 0.33) * v))
            edges = cv.Canny(blurred, low, high, apertureSize=3, L2gradient=True)

            # --- Морфология (CLOSE): «сшиваем» разрывы в линиях ---
            kernel = np.ones((3,3), np.uint8)
            binary = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=1)

            # --- Выбор режима контуров с гистерезисом и удержанием ---
            if mode_hold > 0:
                mode_hold -= 1  # пока таймер не истёк — режим фиксирован
            else:
                # если устойчиво перегруз (avg_bps выше верхнего порога) — включаем EXT
                if mode == 'TREE' and avg_bps > UP_THRESH:
                    mode = 'EXT'
                    mode_hold = 20   # удерживаем новый режим 20 кадров
                # если устойчиво недогруз — возвращаем TREE
                elif mode == 'EXT' and avg_bps < DOWN_THRESH:
                    mode = 'TREE'
                    mode_hold = 20

            retr_mode = cv.RETR_EXTERNAL if mode == 'EXT' else cv.RETR_TREE
            contours, hierarchy = cv.findContours(binary, retr_mode, cv.CHAIN_APPROX_SIMPLE)

            # --- Плавный epsilon для RDP (от нагрузки) ---
            # Нормируем сглаженный битрейт между нижней/верхней границей:
            down = TARGET_BITRATE_BPS * 1.0
            up   = TARGET_BITRATE_BPS * PEAK_FACTOR
            load_ratio = (avg_bps - down) / max(1, (up - down))
            load_ratio = max(0.0, min(1.0, load_ratio))  # clamp в [0..1]

            # Целевой eps от нагрузки (0 → BASE_EPS, 1 → OVER_EPS)
            eps_target = BASE_EPS + (OVER_EPS - BASE_EPS) * load_ratio
            # Сглаживаем eps, чтобы не прыгал между кадрами
            eps_smooth = 0.7 * eps_smooth + 0.3 * eps_target
            eps = eps_smooth

            # --- Фильтруем мусор и упрощаем контуры RDP ---
            filtered_contours = []
            for cnt in contours:
                pts = cnt.reshape(-1, 2).astype(np.float32)  # (N,1,2) → (N,2)
                if pts.shape[0] < 3:
                    continue
                if cv.contourArea(pts.astype(np.int32)) < MIN_AREA:
                    continue
                if cv.arcLength(pts.astype(np.int32), True) < MIN_ARCLEN:
                    continue

                approx = approxPolyDP(pts, epsilon=eps, closed=True)
                if approx.shape[0] >= 3:
                    filtered_contours.append(approx.astype(np.int16))

            # --- Серилизация и сжатие (DEFLATE) ---
            serialized       = pickle.dumps(filtered_contours, protocol=pickle.HIGHEST_PROTOCOL)
            compressed       = zlib.compress(serialized)
            message_size     = struct.pack("Q", len(compressed))          # 8 байт длины
            bytes_this_frame = len(message_size) + len(compressed)        # общий размер кадра

            # --- Отправка ---
            client_socket.sendall(message_size + compressed)
            bytes_sent += bytes_this_frame

            # --- Реальный битрейт за ~1 секунду (скользящее окно) ---
            now = time.time()
            tx_window.append((now, bytes_this_frame))
            tx_bytes_sum += bytes_this_frame
            # выкидываем устаревшие записи старше 1.0 s
            while tx_window and (now - tx_window[0][0]) > 1.0:
                _, old_b = tx_window.popleft()
                tx_bytes_sum -= old_b
            # вычисляем bps = (байты_в_окне * 8) / длительность_окна
            window_secs = max(1e-3, (tx_window[-1][0] - tx_window[0][0]))
            real_bps = (tx_bytes_sum * 8) / window_secs

            # --- EMA: сглаживаем реальный битрейт для решений ---
            avg_bps = (1.0 - EMA_ALPHA) * avg_bps + EMA_ALPHA * real_bps

            # --- Управление FPS и сон до целевого кадра ---
            processing_time = now - frame_start
            if processing_time < frame_delay:
                target_fps = min(30, target_fps + 2)
            else:
                target_fps = max(15, target_fps - 5)
            frame_delay = 1.0 / target_fps
            sleep_time = frame_delay - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)

            # --- Статистика / логи ---
            frame_count   += 1
            total_latency += processing_time
            avg_latency    = total_latency / frame_count
            bitrate_mbps   = real_bps / 1_000_000

            if frame_count % 15 == 0:
                print(f"mode={mode}  eps={eps:.2f}  "
                      f"low/high={low}/{high}  "
                      f"avg_bps={int(avg_bps)}  real_bps={int(real_bps)}  "
                      f"frame={frame.shape[1]}x{frame.shape[0]}")

            print(
                f"Битрейт: {bitrate_mbps:.2f} Мбит/с | "
                f"Задержка: {processing_time:.3f} сек | "
                f"Средняя: {avg_latency:.3f} сек | "
                f"FPS: {target_fps} | "
                f"Контуров: {len(filtered_contours)}"
            )
            bytes_sent = 0

            # --- Предпросмотр (для отладки) ---
            contour_img = np.zeros((target_height, target_width), dtype=np.uint8)
            for cnt in filtered_contours:
                try:
                    reshaped = cnt.reshape(-1, 1, 2).astype(np.int32)
                    if reshaped.shape[0] > 0:
                        cv.drawContours(contour_img, [reshaped], -1, 255, 1)
                except Exception as e:
                    print(f"Ошибка при отрисовке: {e}")

            cv.imshow("Preview", contour_img)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    except (KeyboardInterrupt, socket.error):
        print("Отключение клиента...")
    finally:
        cap.release()
        cv.destroyAllWindows()
        client_socket.close()

if __name__ == "__main__":
    run_client()
