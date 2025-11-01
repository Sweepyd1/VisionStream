import socket
import cv2 as cv
import zlib
import struct
import msgpack
import numpy as np

port = 9999

def run_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", port))
    server_socket.listen(1)
    print(f"📡 Сервер слушает порт {port}...")

    while True:
        try:
            client_socket, addr = server_socket.accept()
            client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client_socket.settimeout(10.0)
            print(f"🔌 Подключён клиент: {addr}")

            data = b""
            payload_size = struct.calcsize("Q")

            while True:
                try:
                    while len(data) < payload_size:
                        packet = client_socket.recv(4096)
                        if not packet:
                            raise ConnectionResetError("Пустой пакет")
                        data += packet

                    packed_msg_size = data[:payload_size]
                    data = data[payload_size:]
                    msg_size = struct.unpack("Q", packed_msg_size)[0]

                    while len(data) < msg_size:
                        packet = client_socket.recv(4096)
                        if not packet:
                            raise ConnectionResetError("Пустой пакет")
                        data += packet

                    frame_data = data[:msg_size]
                    data = data[msg_size:]

                    decompressed = zlib.decompress(frame_data)
                    contours = msgpack.unpackb(decompressed, raw=False)


                    canvas = np.zeros((180, 320), dtype=np.uint8)
                    for cnt in contours:
                        pts = np.array(cnt, dtype=np.int32).reshape((-1, 1, 2))
                        cv.polylines(canvas, [pts], isClosed=True, color=255, thickness=1)

                    cv.imshow("Server: Contours", canvas)
                    if cv.waitKey(1) == ord('q'):
                        break

                    client_socket.sendall(b"ACK")

                except (ConnectionResetError, socket.timeout, struct.error, zlib.error, msgpack.UnpackException) as e:
                    print(f"⚠️ Ошибка обработки кадра: {e}")
                    break

        except KeyboardInterrupt:
            print("\n🛑 Сервер остановлен")
            break
        except Exception as e:
            print(f"❌ Ошибка соединения: {e}")
        finally:
            try:
                client_socket.close()
            except:
                pass
            cv.destroyAllWindows()

    server_socket.close()

if __name__ == "__main__":
    run_server()