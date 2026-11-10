from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import wave
from email import policy
from email.parser import BytesParser

import httpx

from .config import Settings, gpu_inventory, runtime_environment, select_gpu
from .jobs import Cancelled
from .metrics import GPUMemorySampler


class YueBackend:
    """Own one native CUDA worker with explicit residency and memory limits."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.process = None
        self.loaded = False
        self.memory = {}
        self.log = settings.home / "native.log"

    def readiness(self):
        required = [
            self.settings.models / self.settings.model_file,
            self.settings.models / "YuE2-Vae-F32.gguf",
            self.settings.executable,
        ]
        return {
            "ready": all(p.is_file() for p in required)
            and bool(self.settings.gpu_uuid),
            "missing_files": [str(p) for p in required if not p.is_file()],
            "loaded": self.loaded
            and self.process is not None
            and self.process.poll() is None,
            "worker_running": self.process is not None and self.process.poll() is None,
            "gpu_uuid": self.settings.gpu_uuid,
            "quant": self.settings.quant,
            "max_seq": self.settings.max_seq,
            "vae_core": self.settings.vae_core,
            "weight_policy": "resident"
            if self.settings.keep_loaded
            else "strict_stage_eviction",
            "batch_size": 1,
            "memory_budget_gib": self.settings.memory_gib,
            "last_memory": self.memory,
        }

    def unload(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None
        self.loaded = False

    def _ensure(self):
        if self.process is not None and self.process.poll() is None:
            return
        if not self.readiness()["ready"]:
            raise RuntimeError(
                "Run yue-music setup-models and scripts/build-native.cmd first"
            )
        select_gpu(self.settings)
        gpu = next(g for g in gpu_inventory() if g["uuid"] == self.settings.gpu_uuid)
        if int(gpu["total_mib"]) < 12000 or int(gpu["free_mib"]) < 7000:
            raise RuntimeError(
                "RTX 3060 needs at least 7000 MiB free; stop other work on this GPU"
            )
        # Never adopt an unrelated process listening on this port.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", self.settings.native_port))
        env = runtime_environment()
        env["CUDA_VISIBLE_DEVICES"] = self.settings.gpu_uuid
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
        env["PATH"] = cuda_bin + os.pathsep + env.get("PATH", "")
        cmd = [
            str(self.settings.executable),
            "--model",
            str(self.settings.models / self.settings.model_file),
            "--vae",
            str(self.settings.models / "YuE2-Vae-F32.gguf"),
            "--device",
            "CUDA0",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.settings.native_port),
            "--max-seq",
            str(self.settings.max_seq),
            "--max-batch",
            "1",
            "--vae-core",
            str(self.settings.vae_core),
        ]
        if self.settings.keep_loaded:
            cmd.append("--keep-loaded")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with self.log.open("ab") as stream:
            self.process = subprocess.Popen(
                cmd,
                env=env,
                cwd=self.settings.home,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=stream,
                creationflags=flags,
            )
        deadline = time.monotonic() + 45
        with self._client() as client:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"Native worker exited: {self.log.read_text('utf-8', errors='replace')[-2000:]}"
                    )
                try:
                    response = client.get("/health")
                    if response.is_success and response.json().get("status") == "ok":
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
        self.unload()
        raise RuntimeError(f"Native startup timed out; inspect {self.log}")

    def _client(self):
        return httpx.Client(
            base_url=f"http://127.0.0.1:{self.settings.native_port}",
            timeout=15,
            trust_env=False,
        )

    def generate(self, req, job_id, progress, cancelled):
        started = time.monotonic()
        progress("starting_worker", 1)
        self._ensure()
        gpu = next(g for g in gpu_inventory() if g["uuid"] == self.settings.gpu_uuid)
        required_free = 1536 if self.settings.keep_loaded else 7000
        if int(gpu["free_mib"]) < required_free:
            raise RuntimeError(
                f"RTX 3060 has less than {required_free} MiB free; stop competing GPU work"
            )
        sampler = GPUMemorySampler(self.process.pid).start()
        log_start = self.log.stat().st_size
        job_log_start = log_start
        native_id = None
        try:
            with self._client() as client:
                payload = req.model_dump(exclude={"client_request_id"})
                # Explicit single stream; upstream off-mode CFG would double the KV cache.
                payload.update(cfg_scale=1.0, lm_batch_size=1, synth_batch_size=1)
                response = client.post("/synth", json=payload)
                response.raise_for_status()
                native_id = response.json()["id"]
                stage, percent = "planning" if req.cot != "off" else "semantic", 5
                deadline = time.monotonic() + 3600
                while True:
                    if cancelled.is_set():
                        client.post("/job", params={"id": native_id, "cancel": 1})
                        # Eviction/restart also cancels kernels that do not poll immediately.
                        self.unload()
                        raise Cancelled()
                    peak = sampler.result["peak_dedicated_mib"]
                    if peak > self.settings.memory_gib * 1024:
                        self.unload()
                        raise RuntimeError(
                            "Native worker exceeded dedicated VRAM budget; job stopped"
                        )
                    if self.process.poll() is not None:
                        raise RuntimeError(
                            "Native worker exited during generation; inspect native.log"
                        )
                    if time.monotonic() > deadline:
                        self.unload()
                        raise TimeoutError("Native generation exceeded one hour")
                    with self.log.open("rb") as logs:
                        logs.seek(log_start)
                        lines = logs.read().decode("utf-8", errors="replace")
                        log_start = logs.tell()
                    if "[AR] Semantic" in lines:
                        stage, percent = "semantic", 30
                    if "[NAR]" in lines:
                        stage, percent = "synthesizing", 65
                    if "[VAE]" in lines or "[Decode]" in lines:
                        stage, percent = "decoding", 90
                    progress(stage, percent)
                    response = client.get("/job", params={"id": native_id})
                    response.raise_for_status()
                    status = response.json()["status"]
                    if status == "done":
                        response = client.get(
                            "/job", params={"id": native_id, "result": 1}
                        )
                        response.raise_for_status()
                        break
                    if status in {"failed", "cancelled"}:
                        if status == "cancelled":
                            raise Cancelled()
                        raise RuntimeError(
                            f"Native generation failed; inspect {self.log}"
                        )
                    cancelled.wait(0.25)
            progress("saving", 97)
            replay, audio, latents = parse_result(
                response.headers["content-type"], response.content
            )
            ext = ".mp3" if req.output_format == "mp3" else ".wav"
            output = self.settings.home / "outputs" / (job_id + ext)
            output.write_bytes(audio)
            with self.log.open("rb") as logs:
                logs.seek(job_log_start)
                output.with_suffix(".native.log").write_bytes(logs.read())
            output.with_suffix(".replay.json").write_text(
                json.dumps(replay, indent=2), "utf-8"
            )
            output.with_suffix(".abc").write_text(replay.get("abc", ""), "utf-8")
            if latents:
                output.with_suffix(".vae").write_bytes(latents)
            duration = len(replay.get("semantic_tokens", "").split(",")) / 25
            if ext == ".wav":
                with wave.open(str(output)) as wav:
                    duration = wav.getnframes() / wav.getframerate()
                    if (
                        wav.getframerate() != 48000
                        or wav.getnchannels() != 2
                        or not wav.getnframes()
                    ):
                        raise RuntimeError(
                            "Native result is not nonempty 48 kHz stereo audio"
                        )
            self.memory = sampler.stop()
            self.loaded = self.settings.keep_loaded
            self.memory.update(
                gpu_uuid=self.settings.gpu_uuid, native_pid=self.process.pid
            )
            result = {
                "audio_path": str(output),
                "audio_url": f"/v1/jobs/{job_id}/audio",
                "score_path": str(output.with_suffix(".abc")),
                "replay_path": str(output.with_suffix(".replay.json")),
                "native_log_path": str(output.with_suffix(".native.log")),
                "duration_seconds": round(duration, 3),
                "duration_is_ceiling": True,
                "hit_duration_ceiling": duration >= req.duration - 0.08,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "seed": req.seed,
                "lm_seed": req.lm_seed,
                "quant": self.settings.quant,
                "memory": self.memory,
            }
            output.with_suffix(".json").write_text(
                json.dumps({"request": req.model_dump(), "result": result}, indent=2),
                "utf-8",
            )
            return result
        except BaseException:
            self.unload()
            raise
        finally:
            sampler.stop()


def parse_result(content_type, body):
    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: "
        + content_type.encode("ascii")
        + b"\r\nMIME-Version: 1.0\r\n\r\n"
        + body
    )
    replay, audio, latents = None, None, None
    for part in message.iter_parts():
        data = part.get_payload(decode=True)
        if part.get_content_type() == "application/json":
            replay = json.loads(data)
        elif part.get_content_type().startswith("audio/"):
            audio = data
        elif part.get_content_type() == "application/octet-stream":
            latents = data
    if replay is None or not audio:
        raise RuntimeError("Native response did not contain replay metadata and audio")
    return replay, audio, latents
