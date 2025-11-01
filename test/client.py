import numpy as np
import cv2 as cv
import socket
import struct
import time
import msgpack
import zlib
import os

host_ip = "127.0.0.1"  
port = 9999
video_path = "../video.mp4"

if not os.path.exists(video_path):
    raise FileNotFoundError(f"Видео не найдено: {video_path}")

def simplify_contour(cnt, epsilon_factor=0.005):
    """Упрощает контур, сохраняя форму."""
    if len(cnt) < 3:
        return None
    area = cv.contourArea(cnt)
    if area < 10:  # Игнорировать мелкие шумы
        return None
    epsilon = epsilon_factor * cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, epsilon, True)
    if len(approx) < 3:
        return None
    return approx.reshape(-1, 2).tolist()  

def connect_with_retry():
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.connect((host_ip, port))
            print("✅ Подключено к серверу")
            return sock
        except Exception as e:
            print(f"⏳ Ожидание сервера... ({e})")
            time.sleep(1)

def run_client():
    target_fps = 10  
    frame_delay = 1.0 / target_fps
    low_res = (320, 180)  

    client_socket = connect_with_retry()
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть видео")

    bytes_sent_total = 0
    total_latency = 0
    frame_count = 0

    try:
        while True:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Видео закончилось")
                break

            
            small = cv.resize(frame, low_res, interpolation=cv.INTER_AREA)
            gray = cv.cvtColor(small, cv.COLOR_BGR2GRAY)
            blurred = cv.GaussianBlur(gray, (7, 7), 0)
            edges = cv.Canny(blurred, 20, 100)
            contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            
            contours = sorted(contours, key=cv.contourArea, reverse=True)[:20]
            simplified = []
            for cnt in contours:
                simp = simplify_contour(cnt)
                if simp:
                    simplified.append(simp)

            
            try:
                serialized = msgpack.packb(simplified, use_bin_type=True)
            except Exception as e:
                print(f"Ошибка сериализации: {e}")
                continue

            compressed = zlib.compress(serialized, level=6)  
            message = struct.pack("Q", len(compressed)) + compressed

            try:
                client_socket.sendall(message)
                bytes_sent_total += len(message)
            except Exception as e:
                print(f"Ошибка отправки: {e}")
                break

            processing_time = time.time() - frame_start

            
            if processing_time < frame_delay:
                target_fps = min(15, target_fps + 0.5)
            else:
                target_fps = max(5, target_fps - 0.5)
            frame_delay = 1.0 / max(1, target_fps)

            
            frame_count += 1
            total_latency += processing_time
            avg_latency = total_latency / frame_count
            bitrate_mbps = (bytes_sent_total * 8) / (frame_count * 1_000_000) * target_fps

            print(
                f"Битрейт: {bitrate_mbps:.3f} Мбит/с | "
                f"Задержка: {processing_time:.3f} с | "
                f"FPS: {target_fps:.1f} | "
                f"Контуров: {len(simplified)}"
            )

            
            preview = np.zeros((low_res[1], low_res[0]), dtype=np.uint8)
            for cnt in simplified:
                pts = np.array(cnt, dtype=np.int32).reshape((-1, 1, 2))
                cv.polylines(preview, [pts], isClosed=True, color=255, thickness=1)
            cv.imshow("Client Preview", frame)
            if cv.waitKey(1) == ord('q'):
                break

            
            sleep_time = max(0, frame_delay - processing_time)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n🛑 Клиент остановлен пользователем")
    finally:
        cap.release()
        cv.destroyAllWindows()
        client_socket.close()

if __name__ == "__main__":
    run_client()