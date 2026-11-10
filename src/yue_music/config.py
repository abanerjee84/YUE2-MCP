from __future__ import annotations

import json
import os
import secrets
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    home: Path
    port: int = 8766
    native_port: int = 8767
    quant: str = "Q8_0"
    max_seq: int = 16384
    vae_core: int = 256
    keep_loaded: bool = True
    gpu_uuid: str = ""
    memory_gib: float = 10.5
    token: str = ""

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def models(self) -> Path:
        return self.home / "models"

    @property
    def model_file(self):
        if self.quant not in {"Q8_0", "Q6_K"}:
            raise ValueError("quant must be Q8_0 or Q6_K")
        return f"YuE2-3B-{self.quant}.gguf"

    @property
    def executable(self):
        return ROOT / "vendor/yue2.cpp/build/yue-server.exe"

    def save(self):
        self.home.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data.pop("home")
        temp = self.home / "config.json.tmp"
        temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temp.replace(self.home / "config.json")


def runtime_environment():
    """Restore public Windows paths stripped by restricted stdio MCP launchers."""
    env = os.environ.copy()
    if os.name == "nt":
        import winreg

        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion"
            ) as key:
                program_files = winreg.QueryValueEx(key, "ProgramFilesDir")[0]
        except OSError:
            program_files = env.get("SYSTEMDRIVE", "C:") + r"\Program Files"
        # NVML fails to initialize on this driver when neither is present.
        env.setdefault("PROGRAMFILES", program_files)
        env.setdefault("PROGRAMW6432", program_files)
    return env


def gpu_inventory() -> list[dict]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
            env=runtime_environment(),
        )
        return [
            dict(
                zip(
                    ("uuid", "name", "total_mib", "free_mib"),
                    (x.strip() for x in line.split(",")),
                    strict=True,
                )
            )
            for line in result.stdout.splitlines()
            if line.strip()
        ]
    except (OSError, subprocess.SubprocessError):
        return []


def load_settings() -> Settings:
    home = Path(os.environ.get("YUE_MUSIC_HOME", ROOT / "data")).resolve()
    home.mkdir(parents=True, exist_ok=True)
    # All processes share this lock, including simultaneous first-use adapters.
    from filelock import FileLock

    with FileLock(str(home / "config.lock")):
        path = home / "config.json"
        data = json.loads(path.read_text("utf-8")) if path.exists() else {}
        settings = Settings(home=home, **data)
        if not settings.token:
            settings.token = secrets.token_urlsafe(32)
        if not settings.gpu_uuid:
            matches = [g for g in gpu_inventory() if "RTX 3060" in g["name"]]
            if len(matches) == 1:
                settings.gpu_uuid = matches[0]["uuid"]
        settings.save()
        return settings


def select_gpu(settings: Settings):
    """Set the child CUDA mask before native startup; never use another GPU."""
    matches = [g for g in gpu_inventory() if g["uuid"] == settings.gpu_uuid]
    if len(matches) != 1 or "RTX 3060" not in matches[0]["name"]:
        raise RuntimeError(
            "Configured RTX 3060 UUID is unavailable. Run yue-music doctor."
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = settings.gpu_uuid
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
