import socket
import time
import cv2 as cv
import struct
import numpy as np
import msgpack
import lz4.frame

port = 9999

def run_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_ip = "192.168.2.148"

    server_socket.bind((host_ip, port))
    server_socket.listen(1)
    print("Сервер запущен!")

    client_socket, addr = server_socket.accept()
    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)

    data = b""
    payload_size = struct.calcsize("Q")

    try:
        while True:
            # ждём заголовок (размер пакета)
            while len(data) < payload_size:
                packet = client_socket.recv(49152)
                if not packet:
                    raise ConnectionResetError("Клиент отключился")
                data += packet

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]

            # ждём весь пакет
            while len(data) < msg_size:
                packet = client_socket.recv(49152)
                if not packet:
                    raise ConnectionResetError("Клиент отключился")
                data += packet

            frame_data = data[:msg_size]
            data = data[msg_size:]

            # --- декодирование ---
            try:
                decompress_start = time.time()
                decompressed = lz4.frame.decompress(frame_data)
                contours = msgpack.unpackb(decompressed, raw=False)
                contours = [np.array(cnt, dtype=np.int16) for cnt in contours]
                decompress_delay = time.time() - decompress_start

                # --- визуализация ---
                canvas = np.zeros((720, 480), dtype=np.uint8)
                for cnt in contours:
                    if (
                        cnt is not None
                        and isinstance(cnt, np.ndarray)
                        and cnt.ndim == 2
                        and cnt.shape[0] >= 1
                        and cnt.shape[1] == 2
                    ):
                        try:
                            reshaped = cnt.reshape(-1, 1, 2).astype(np.int32)
                            if reshaped.shape[0] > 0:
                                cv.drawContours(canvas, [reshaped], -1, 255, 1)
                        except Exception as e:
                            print(f"Ошибка отрисовки контура: {e}")

                cv.imshow("Контуры (сервер)", canvas)
                print(f"Декодирование контуров: {decompress_delay:.4f} сек | Контуров: {len(contours)}")

            except Exception as e:
                print(f"Ошибка обработки: {e}")

            if cv.waitKey(1) == ord("q"):
                break

    except (ConnectionResetError, BrokenPipeError):
        print("Соединение закрыто клиентом")

    finally:
        client_socket.close()
        server_socket.close()
        cv.destroyAllWindows()

if __name__ == "__main__":
    run_server()
