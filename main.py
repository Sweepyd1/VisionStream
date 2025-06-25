import sys
from src.client import run_client, run_client_v2, run_client_v3
from src.server import run_server, run_server_v2


def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        versions = sys.argv[2]
        if mode == "client":
            if versions == 1:
                run_client()
            if versions == 2:
                run_client_v2()
            if versions == 3:
                run_client_v3()
        elif mode == "server":
            if versions == 1:
                run_server()
            if versions == 2:
                run_server_v2()
        else:
            print("такого режима запуска нет!")
    else:
        mode = None


if __name__ == "__main__":
    main()
