import numpy as np
import cv2 as cv
import socket

import struct
from config import host_ip, port
import time

port = int(port)


def run_client():
    frame_rate = 25  # Задание желаемой частоты кадров (25 кадров в секунду)
    frame_delay = (
        1.0 / frame_rate
    )  # Вычисление задержки между кадрами для поддержания частоты

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(
        socket.IPPROTO_TCP, socket.TCP_NODELAY, 1
    )  # Отключение алгоритма Нейгла для уменьшения задержек

    client_socket.connect((host_ip, port))  # Подключение к серверу по IP и порту

    cap = cv.VideoCapture(
        0
    )  
    cap.set(
        cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc("M", "J", "P", "G")
    )  # Установка кодека захвата MJPG для повышения скорости
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)  
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)  

    try:
        while True:  
            proc_start = time.time()  

            ret, frame = cap.read()  
            if  not ret:
                break

            gray = cv.cvtColor(
                frame, cv.COLOR_BGR2GRAY
            )  
            edges = cv.Canny(gray, 100, 200)  # Применяем детектор границ Canny
            contours, _ = cv.findContours(
                edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE
            )  # Находим контуры на изображении
            contour_img = np.zeros(
                (gray.shape[0], gray.shape[1], 3), dtype=np.uint8
            )  # Создаем пустое цветное изображение для отрисовки контуров
            cv.drawContours(
                contour_img, contours, -1, (255, 255, 255), 1
            )  # Рисуем все контуры белым цветом толщиной 1 пиксель

            encode_param = [
                int(cv.IMWRITE_JPEG_QUALITY),
                30,
            ]  # Параметры сжатия JPEG с качеством 30 (низкое качество для меньшего размера)
            _, jpeg_frame = cv.imencode(
                ".jpg", contour_img, encode_param
            )  # Кодируем изображение с контурами в JPEG формат
            data = (
                jpeg_frame.tobytes()
            )  # Преобразуем сжатое изображение в байтовый поток

            proc_end = time.time()  

            processing_delay = (
                proc_end - proc_start
            )  

            message_size = struct.pack(
                "Q", len(data)
            )  
            send_start = time.time()  
            client_socket.sendall(
                message_size + data
            )  
            ack = client_socket.recv(
                1024
            )  # Ожидаем подтверждение от сервера 
            send_delay = (
                time.time() - send_start
            )  # Вычисляем время, затраченное на отправку и получение подтверждения

            elapsed = time.time() - proc_start  # Общее время с начала обработки кадра
            if (
                elapsed < frame_delay
            ):  # Если прошло меньше времени, чем нужно для заданной частоты кадров
                time.sleep(
                    frame_delay - elapsed
                )  # Ждем оставшееся время, чтобы поддерживать стабильную частоту кадров

            rtt = (
                time.time() - send_start
            )  # Рассчитываем время — задержку подтверждения

            data_size_bits = len(data) * 8  
            bitrate_mbps = (
                (data_size_bits / send_delay) / 1_000_000
            )  

            print(
                f"Битрейт (расчёт по времени отправки): {bitrate_mbps:.2f} Мбит/с | "
                f"Время обработки (обработка + сжатие): {processing_delay:.4f} сек | "
                f"Время передачи (sendall): {send_delay:.4f} сек | "
                f"RTT (задержка подтверждения): {rtt:.4f} сек"
            )

            cv.imshow(
                "Client Preview", contour_img
            )  

            if cv.waitKey(1) == ord("q"):  
                break
    except KeyboardInterrupt:  
        client_socket.close()  
    finally:
        cap.release()  
        cv.destroyAllWindows()  # Закрываем все окна OpenCV
        client_socket.close()  # Закрываем сокет клиента (на всякий случай, если не закрыт ранее)
