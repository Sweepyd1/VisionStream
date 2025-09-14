import numpy as np
import cv2 as cv
import socket
import struct
import time


host_ip = "192.168.2.124"
port = 9999


# def run_client():
#     bytes_sent = 0

#     target_fps = 20
#     frame_delay = 1.0 / target_fps

#     target_width, target_height = 600, 480

#     client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
#     client_socket.connect((host_ip, port))

#     cap = cv.VideoCapture(0)

#     cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc("M", "J", "P", "G"))
#     cap.set(cv.CAP_PROP_FRAME_WIDTH, target_width)
#     cap.set(cv.CAP_PROP_FRAME_HEIGHT, target_height)
#     cap.set(cv.CAP_PROP_FPS, target_fps)

#     contour_img = np.zeros((target_height, target_width), dtype=np.uint8)

#     try:
#         total_latency = 0
#         frame_count = 0

#         while True:
#             frame_start = time.time()
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             gray = frame[:, :, 0]

#             _, binary = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)

#             contours, _ = cv.findContours(
#                 binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
#             )

#             contour_img.fill(0)

#             for cnt in contours:
#                 if cv.contourArea(cnt) > 100:
#                     cv.drawContours(contour_img, [cnt], -1, 255, 1)

#             # Используем быстрое сжатие JPEG с низким качеством
#             encode_param = [int(cv.IMWRITE_JPEG_QUALITY), 30]
#             _, jpeg_frame = cv.imencode(".jpg", contour_img, encode_param)
#             data = jpeg_frame.tobytes()

#             message_size = struct.pack("Q", len(data))
#             client_socket.sendall(message_size + data)
#             bytes_sent += len(message_size) + len(data)

#             processing_time = time.time() - frame_start

#             if processing_time < frame_delay:
#                 target_fps = min(30, target_fps + 2)
#             else:
#                 target_fps = max(15, target_fps - 5)

#             frame_delay = 1.0 / target_fps

#             frame_count += 1
#             total_latency += processing_time
#             avg_latency = total_latency / frame_count
#             bitrate_mbps = bytes_sent * 8 / 1_000_000
#             print(
#                 f"Битрейт: {bitrate_mbps:.2f} Мбит/с | "
#                 f"Задержка: {processing_time:.3f} сек | "
#                 f"Средняя: {avg_latency:.3f} сек | "
#                 f"FPS: {target_fps} | "
#                 f"Контуров: {len(contours)}"
#             )
#             bytes_sent = 0

#             cv.imshow("Preview", contour_img)
#             cv.waitKey(1)


#     except (KeyboardInterrupt, socket.error):
#         pass
#     finally:
#         cap.release()
#         cv.destroyAllWindows()
#         client_socket.close()
def run_client(video_file):
    bytes_sent = 0
    target_fps = 20
    frame_delay = 1.0 / target_fps
    target_width, target_height = 600, 480

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    client_socket.connect((host_ip, port))

    # Открываем видеофайл вместо камеры
    cap = cv.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Ошибка открытия видеофайла: {video_file}")
        return

    contour_img = np.zeros((target_height, target_width), dtype=np.uint8)

    try:
        total_latency = 0
        frame_count = 0

        while True:
            frame_start = time.time()
            ret, frame = cap.read()

            # Выход при окончании видео
            if not ret:
                break

            # Изменение размера кадра
            frame = cv.resize(frame, (target_width, target_height))
            gray = frame[:, :, 0]

            _, binary = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)
            contours, _ = cv.findContours(
                binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )

            contour_img.fill(0)
            for cnt in contours:
                if cv.contourArea(cnt) > 100:
                    cv.drawContours(contour_img, [cnt], -1, 255, 1)

            # Сжатие JPEG
            encode_param = [int(cv.IMWRITE_JPEG_QUALITY), 30]
            _, jpeg_frame = cv.imencode(".jpg", contour_img, encode_param)
            data = jpeg_frame.tobytes()

            # Отправка данных
            message_size = struct.pack("Q", len(data))
            client_socket.sendall(message_size + data)
            bytes_sent += len(message_size) + len(data)

            # Регулировка FPS
            processing_time = time.time() - frame_start
            sleep_time = frame_delay - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Динамическая корректировка FPS
            if processing_time < frame_delay:
                target_fps = min(30, target_fps + 2)
            else:
                target_fps = max(15, target_fps - 5)
            frame_delay = 1.0 / target_fps

            # Статистика
            frame_count += 1
            total_latency += processing_time
            avg_latency = total_latency / frame_count
            bitrate_mbps = bytes_sent * 8 / 1_000_000
            print(
                f"Битрейт: {bitrate_mbps:.2f} Мбит/с | "
                f"Задержка: {processing_time:.3f} сек | "
                f"Средняя: {avg_latency:.3f} сек | "
                f"FPS: {target_fps} | "
                f"Контуров: {len(contours)}"
            )
            bytes_sent = 0

            # Отображение превью (опционально)
            # cv.imshow("Preview", contour_img)
            # if cv.waitKey(1) & 0xFF == ord('q'):
            #     break

    except (KeyboardInterrupt, socket.error) as e:
        print(f"Ошибка: {e}")
    # finally:
    #     cap.release()
    #     cv.destroyAllWindows()
    #     client_socket.close()


def run_client_v2():
    total_bytes = 0
    last_report_time = time.time()
    frame_count = 0
    target_fps = 25
    frame_delay = 1.0 / target_fps

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    client_socket.connect((host_ip, port))

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv.CAP_PROP_FPS, target_fps)

    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gaussian_kernel = 5
    contour_img = np.zeros((720, 1280), dtype=np.uint8)

    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Гауссово размытие для уменьшения шума
            blurred = cv.GaussianBlur(gray, (gaussian_kernel, gaussian_kernel), 0)

            # Детектор границ Кэнни для получения непрерывных линий
            edges = cv.Canny(blurred, 50, 150)

            # Морфологические операции для сглаживания границ
            dilated = cv.dilate(edges, kernel, iterations=1)
            smoothed = cv.morphologyEx(dilated, cv.MORPH_CLOSE, kernel)

            # Поиск и аппроксимация контуров
            contours, _ = cv.findContours(
                smoothed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
            )

            contour_img.fill(0)
            for contour in contours:
                # Аппроксимация контура для сглаживания
                epsilon = 0.005 * cv.arcLength(contour, True)
                approx = cv.approxPolyDP(contour, epsilon, True)
                cv.drawContours(contour_img, [approx], -1, 255, 1)

            small_contour = cv.resize(contour_img, (480, 270))

            quality = 10
            _, jpeg_frame = cv.imencode(
                ".jpg", small_contour, [int(cv.IMWRITE_JPEG_QUALITY), quality]
            )

            target_size = 1000
            step = 0
            while len(jpeg_frame) > target_size and step < 5:
                quality = max(5, quality - 5)
                _, jpeg_frame = cv.imencode(
                    ".jpg", small_contour, [int(cv.IMWRITE_JPEG_QUALITY), quality]
                )
                step += 1

            data = jpeg_frame.tobytes()
            frame_size = len(data)
            header = struct.pack("Q", frame_size)

            client_socket.sendall(header + data)
            total_bytes += len(header) + frame_size
            frame_count += 1

            current_time = time.time()
            elapsed = current_time - last_report_time
            if elapsed >= 1.0:
                bitrate = (total_bytes * 8) / elapsed / 1_000_000
                actual_fps = frame_count / elapsed
                print(
                    f"Битрейт: {bitrate:.3f} Мбит/с | "
                    f"FPS: {actual_fps:.1f}/{target_fps} | "
                    f"Размер: {frame_size} байт | "
                    f"Качество: {quality} | "
                    f"Контуры: {len(contours)}"
                )
                total_bytes = 0
                frame_count = 0
                last_report_time = current_time

            processing_time = current_time - start_time
            sleep_time = max(0, frame_delay - processing_time)
            time.sleep(sleep_time)

            cv.imshow("Preview", contour_img)
            if cv.waitKey(1) == 27:
                break

    except (KeyboardInterrupt, socket.error) as e:
        print(f"Ошибка: {e}")
    finally:
        cap.release()
        cv.destroyAllWindows()
        client_socket.close()
