import argparse
import json
import logging
import sys
from pathlib import Path

from .client import Client
from .config import gpu_inventory, load_settings
from .schema import MusicRequest


def emit(value):
    print(json.dumps(value, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        prog="yue-music", description="Local YuE2 GGUF music tools"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("start", "stop", "doctor", "mcp", "install-cli"):
        sub.add_parser(command)
    serve = sub.add_parser("serve")
    serve.add_argument("--background", action="store_true", help=argparse.SUPPRESS)
    setup = sub.add_parser("setup-models")
    setup.add_argument(
        "--both", action="store_true", help="Download Q8 and Q6 for comparison"
    )
    register = sub.add_parser("register")
    register.add_argument("harness", choices=["codex", "claude-code", "gemini"])
    cfg = sub.add_parser("mcp-config")
    cfg.add_argument("--http", action="store_true")
    configure = sub.add_parser("configure")
    configure.add_argument("--quant", choices=["Q8_0", "Q6_K"])
    configure.add_argument("--max-seq", type=int, choices=[8192, 16384, 24576])
    configure.add_argument("--residency", choices=["resident", "staged"])
    gen = sub.add_parser("generate")
    inputs = gen.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--style")
    inputs.add_argument("--request", type=Path, help="MusicRequest JSON, UTF-8")
    lyrics = gen.add_mutually_exclusive_group()
    lyrics.add_argument("--lyrics", default="")
    lyrics.add_argument("--lyrics-file", type=Path)
    gen.add_argument("--abc-file", type=Path)
    gen.add_argument("--duration", type=float, default=60)
    gen.add_argument("--cot", choices=["full", "melody", "off"], default="full")
    gen.add_argument("--steps", type=int, default=8)
    gen.add_argument(
        "--seed", type=int, help="Use -1 or omit for a random acoustic seed"
    )
    gen.add_argument(
        "--lm-seed",
        type=int,
        help="Omit to inherit --seed; use -1 for an independent random token seed",
    )
    gen.add_argument("--format", choices=["wav16", "mp3"], default="wav16")
    gen.add_argument("--request-id")
    gen.add_argument("--wait", action="store_true")
    gen.add_argument("--timeout", type=float, default=3600)
    gen.add_argument("--out", type=Path)
    for command in ("status", "cancel", "wait", "download"):
        cmd = sub.add_parser(command)
        cmd.add_argument("job_id")
        if command == "wait":
            cmd.add_argument("--timeout", type=float, default=3600)
        if command == "download":
            cmd.add_argument("--out", type=Path, required=True)
            cmd.add_argument(
                "--artifact", choices=["audio", "score", "replay"], default="audio"
            )
    args = parser.parse_args()
    settings = load_settings()
    client = Client(settings)
    try:
        if args.command == "serve":
            if args.background:
                log = (settings.home / "service.log").open(
                    "a", encoding="utf-8", buffering=1
                )
                sys.stdout = sys.stderr = log
            import uvicorn
            from filelock import FileLock

            from .service import create_app

            with FileLock(str(settings.home / "worker.lock"), timeout=0):
                logging.basicConfig(level=logging.INFO)
                app = create_app(settings)
                server = uvicorn.Server(
                    uvicorn.Config(
                        app,
                        host="127.0.0.1",
                        port=settings.port,
                        workers=1,
                        log_level="info",
                    )
                )
                app.state.shutdown = lambda: setattr(server, "should_exit", True)
                server.run()
        elif args.command == "mcp":
            from .mcp_adapter import create_mcp

            create_mcp(settings).run(transport="stdio")
        elif args.command == "start":
            emit(client.health())
        elif args.command == "stop":
            emit(
                client._request("POST", "/admin/shutdown").json()
                if client.reachable()
                else {"status": "not_running"}
            )
        elif args.command == "doctor":
            from .backend import YueBackend

            running = client.reachable()
            emit(
                {
                    "home": str(settings.home),
                    "gpus": gpu_inventory(),
                    "service_running": running,
                    "inference": client.health()["inference"]
                    if running
                    else YueBackend(settings).readiness(),
                }
            )
        elif args.command == "setup-models":
            from .models import setup_models

            emit(setup_models(settings, both=args.both))
        elif args.command == "configure":
            if client.reachable():
                raise RuntimeError(
                    "Stop the service before changing the inference profile"
                )
            if args.quant:
                settings.quant = args.quant
            if args.max_seq:
                settings.max_seq = args.max_seq
            if args.residency:
                settings.keep_loaded = args.residency == "resident"
            settings.save()
            emit(
                {
                    "quant": settings.quant,
                    "max_seq": settings.max_seq,
                    "keep_loaded": settings.keep_loaded,
                }
            )
        elif args.command in {"install-cli", "register", "mcp-config"}:
            from .install import install_cli, mcp_entry, register_codex, register_json

            if args.command == "install-cli":
                emit(install_cli(settings))
            elif args.command == "register":
                emit(
                    register_codex(settings)
                    if args.harness == "codex"
                    else register_json(settings, args.harness)
                )
            elif args.http:
                emit(
                    {
                        "url": settings.url + "/mcp",
                        "authentication": "Bearer token in data/config.json",
                    }
                )
            else:
                emit({"mcpServers": {"local-yue": mcp_entry(settings)}})
        elif args.command == "generate":
            if args.request:
                req = MusicRequest.model_validate_json(args.request.read_text("utf-8"))
            else:
                req = MusicRequest(
                    style=args.style,
                    lyrics=args.lyrics_file.read_text("utf-8")
                    if args.lyrics_file
                    else args.lyrics,
                    abc=args.abc_file.read_text("utf-8") if args.abc_file else "",
                    cot=args.cot,
                    duration=args.duration,
                    steps=args.steps,
                    seed=args.seed,
                    lm_seed=args.lm_seed,
                    output_format=args.format,
                    client_request_id=args.request_id,
                )
            job = client.submit(req.model_dump())
            if args.wait or args.out:
                job = client.wait(job["job_id"], args.timeout)
            if args.out and job["status"] == "succeeded":
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_bytes(client.artifact(job["job_id"]))
                job["saved_copy"] = str(args.out.resolve())
            emit(job)
            return 1 if job["status"] in {"failed", "cancelled"} else 0
        elif args.command == "status":
            emit(client.job(args.job_id))
        elif args.command == "cancel":
            emit(client.cancel(args.job_id))
        elif args.command == "wait":
            job = client.wait(args.job_id, args.timeout)
            emit(job)
            return 0 if job["status"] == "succeeded" else 1
        elif args.command == "download":
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_bytes(client.artifact(args.job_id, args.artifact))
            emit({"saved": str(args.out.resolve())})
        return 0
    except Exception as exc:
        emit({"error": f"{type(exc).__name__}: {exc}"})
        return 1
