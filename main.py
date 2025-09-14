import sys
from src.client import run_client, run_client_v2
from src.server import run_server
file = "./src/video.mp4"

def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        versions = sys.argv[2]
        if mode == "client":
            if versions == "1":
                run_client(video_file=file)
            if versions == "2":
                run_client_v2()
        elif mode == "server":
            if versions == "1":
                run_server()
        else:
            print("такого режима запуска нет!")
    else:
        mode = None

if __name__ == "__main__":
    main()
