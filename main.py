import sys
from src.client import run_client, run_client_v2
from src.server import run_server


def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "client":
            run_client()
        elif mode == "server":
            run_server()
        else:
            print("такого режима запуска нет!")
    else:
        mode = None


if __name__ == "__main__":
    main()
