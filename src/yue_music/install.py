from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

from .config import ROOT, Settings


def mcp_entry(settings: Settings):
    return {
        "command": str(Path(sys.executable).resolve()),
        "args": ["-m", "yue_music", "mcp"],
        "env": {"YUE_MUSIC_HOME": str(settings.home)},
    }


def install_cli(settings: Settings):
    target = ROOT / "bin"
    target.mkdir(exist_ok=True)
    if os.name == "nt":
        wrapper = target / "yue-music.cmd"
        wrapper.write_text(
            f'@echo off\r\n"{sys.executable}" -m yue_music %*\r\n', encoding="utf-8"
        )
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            "Environment",
            0,
            winreg.KEY_READ | winreg.KEY_WRITE,
        ) as key:
            try:
                existing, kind = winreg.QueryValueEx(key, "Path")
            except FileNotFoundError:
                existing, kind = "", winreg.REG_EXPAND_SZ
            entries = [p for p in existing.split(";") if p]
            if str(target).casefold() not in [
                p.rstrip("\\/").casefold() for p in entries
            ]:
                entries.append(str(target))
                winreg.SetValueEx(key, "Path", 0, kind, ";".join(entries))
        # Notify shells launched from Explorer of the user PATH change.
        import ctypes

        result = ctypes.c_size_t()
        ctypes.windll.user32.SendMessageTimeoutW(
            0xFFFF, 0x001A, 0, "Environment", 2, 1000, ctypes.byref(result)
        )
    else:
        wrapper = target / "yue-music"
        import shlex

        wrapper.write_text(
            f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -m yue_music "$@"\n'
        )
        wrapper.chmod(0o755)
    return {
        "launcher": str(wrapper),
        "path_directory": str(target),
        "note": "Open a new terminal to receive the updated PATH.",
    }


def register_codex(settings: Settings, path: Path | None = None):
    import tomlkit

    path = (
        path
        or Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "config.toml"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    from filelock import FileLock

    with FileLock(str(path) + ".yue-music.lock"):
        doc = (
            tomlkit.parse(path.read_text("utf-8"))
            if path.exists()
            else tomlkit.document()
        )
        entry = mcp_entry(settings)
        entry.update(startup_timeout_sec=30, tool_timeout_sec=60)
        servers = doc.setdefault("mcp_servers", tomlkit.table())
        if "local-yue" in servers and dict(servers["local-yue"]) != entry:
            raise RuntimeError(
                "An existing local-yue MCP entry differs; refusing to replace it"
            )
        if "local-yue" not in servers:
            if path.exists():
                backup = path.with_name(path.name + f".yue-music-{time.time_ns()}.bak")
                shutil.copy2(path, backup)
            servers["local-yue"] = entry
            temp = path.with_name(path.name + ".yue-music.tmp")
            temp.write_text(tomlkit.dumps(doc), encoding="utf-8")
            temp.replace(path)
    return {
        "registered": "local-yue",
        "config": str(path),
        "note": "Restart the harness to discover these tools.",
    }


def register_json(settings: Settings, harness: str, path: Path | None = None):
    targets = {
        "claude-code": Path.home() / ".claude.json",
        "gemini": Path.home() / ".gemini/settings.json",
    }
    path = path or targets[harness]
    entry = mcp_entry(settings)
    if harness == "claude-code":
        entry["type"] = "stdio"
    path.parent.mkdir(parents=True, exist_ok=True)
    from filelock import FileLock

    with FileLock(str(path) + ".yue-music.lock"):
        doc = json.loads(path.read_text("utf-8")) if path.is_file() else {}
        servers = doc.setdefault("mcpServers", {})
        if "local-yue" in servers and servers["local-yue"] != entry:
            raise RuntimeError(
                "An existing local-yue MCP entry differs; refusing to replace it"
            )
        if "local-yue" not in servers:
            if path.exists():
                shutil.copy2(
                    path, path.with_name(path.name + f".yue-music-{time.time_ns()}.bak")
                )
            servers["local-yue"] = entry
            temp = path.with_name(path.name + ".yue-music.tmp")
            temp.write_text(
                json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            temp.replace(path)
    return {
        "registered": "local-yue",
        "harness": harness,
        "config": str(path),
        "note": "Restart the harness to discover these tools.",
    }
