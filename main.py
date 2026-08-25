import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Clarity Video AI", add_help=False)
    parser.add_argument("--cli", action="store_true", help="Launch interactive terminal CLI instead of Web UI")
    parser.add_argument("--web", action="store_true", help="Launch Studio Web UI (default)")
    parser.add_argument("--host", default="127.0.0.1", help="Host IP for Web UI")
    parser.add_argument("--port", type=int, default=7860, help="Port for Web UI")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")

    cli_flags = {"--cli", "--download-models", "-h", "--help", "benchmark"}
    argv_set = set(sys.argv[1:])
    if cli_flags.intersection(argv_set) and not ("--web" in argv_set):
        from video_upscaler.cli import run as run_cli
        run_cli()
        return

    args, _ = parser.parse_known_args()
    if args.cli:
        from video_upscaler.cli import run as run_cli
        run_cli()
    else:
        from video_upscaler.web.server import run_server
        run_server(host=args.host, port=args.port, open_browser=not args.no_browser)

if __name__ == "__main__":
    main()
