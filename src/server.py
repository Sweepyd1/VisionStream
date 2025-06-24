import socket
import time
import cv2

import struct

import numpy as np
from config import port

port = int(port)


def run_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
    host_ip = "0.0.0.0"  

    server_socket.bind((host_ip, port))  
    server_socket.listen(1)  
    print("сервер запущен!") 

    client_socket, addr = server_socket.accept() 
    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Отключаем алгоритм Нейгла для уменьшения задержек
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)  # Устанавливаем размер буфера приема в 1 МБ

    data = b""  
    payload_size = struct.calcsize("Q")  

    try:
        while True:  
            
            while len(data) < payload_size:
                packet = client_socket.recv(49152)  
                if not packet:  
                    print("Клиент отключился")
                    raise ConnectionResetError
                data += packet  

            packed_msg_size = data[:payload_size]  # Извлекаем первые 8 байт — размер сообщения
            data = data[payload_size:]  # Удаляем из буфера прочитанные байты
            msg_size = struct.unpack("Q", packed_msg_size)[0]  # Распаковываем длину сообщения

            while len(data) < msg_size:
                packet = client_socket.recv(49152)  
                if not packet:  
                    print("Клиент отключился")
                    raise ConnectionResetError
                data += packet  

            frame_data = data[:msg_size]  # Извлекаем байты кадра из буфера
            data = data[msg_size:]  

            client_socket.sendall(b"ACK")  # Отправляем клиенту подтверждение получения кадра

            try:
                decode_start = time.time()  # Запоминаем время начала декодирования
                # Декодируем JPEG-байты в изображение OpenCV
                frame = cv2.imdecode(
                    np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                frame = cv2.resize(frame, (1280, 720))
                decode_delay = time.time() - decode_start  # Время декодирования

                
                if frame.shape == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                cv2.imshow("Server Display", frame)  
                print(f"Декодирование: {decode_delay:.4f} сек")  
            except Exception as e:  
                print(f"Ошибка декодирования: {e}")

            if cv2.waitKey(1) == ord("q"):  
                break

    except (ConnectionResetError, BrokenPipeError):  
        print("Соединение закрыто клиентом")

    finally:
        client_socket.close()  
        server_socket.close()  
        cv2.destroyAllWindows()  