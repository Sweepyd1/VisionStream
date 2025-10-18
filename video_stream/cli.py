import argparse
from video_stream.client import VideoClient
from video_stream.server import VideoServer

def main():
    parser = argparse.ArgumentParser(description="Видео-стриминг: клиент или сервер")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Клиент
    client_parser = subparsers.add_parser("client", help="Запустить клиент")
    client_parser.add_argument("--host", default="127.0.0.1", help="IP сервера")
    client_parser.add_argument("--port", type=int, default=9999, help="Порт сервера")
    client_parser.add_argument("--source", default="0", help="Источник: 0 (камера) или путь к файлу")
    client_parser.add_argument("--version", choices=["1", "2"], default="1", help="Версия обработки")

    # Сервер
    server_parser = subparsers.add_parser("server", help="Запустить сервер")
    server_parser.add_argument("--host", default="0.0.0.0", help="Хост для прослушивания")
    server_parser.add_argument("--port", type=int, default=9999, help="Порт")

    args = parser.parse_args()

    if args.mode == "client":
        use_advanced = args.version == "2"
        client = VideoClient(
            host=args.host,
            port=args.port,
            use_advanced_processing=use_advanced,
            report_interval=1.0 if use_advanced else None,
            target_compression_size=1000 if use_advanced else None,
            initial_quality=10 if use_advanced else 30,
            target_width=1280 if use_advanced else 600,
            target_height=720 if use_advanced else 480,
            target_fps=25 if use_advanced else 20,
        )
        source = int(args.source) if args.source.isdigit() else args.source
        client.run(source)

    elif args.mode == "server":
        server = VideoServer(host=args.host, port=args.port)
        server.run()

if __name__ == "__main__":
    main()