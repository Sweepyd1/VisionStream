import numpy as np
import cv2 as cv
import socket
import pickle
import struct
from config import host_ip, port

port = int(port)

def run_client():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host_ip, port))
    cap = cv.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            edges = cv.Canny(gray, 100, 200)
            contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            contour_img = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
            cv.drawContours(contour_img, contours, -1, (255, 255, 255), 1)

            data = pickle.dumps(contour_img)
            message_size = struct.pack("Q", len(data))
            client_socket.sendall(message_size + data)

            cv.imshow("Client Preview", contour_img)
            if cv.waitKey(1) == ord("q"):
                break

    finally:
        cap.release()
        cv.destroyAllWindows()
        client_socket.close()
