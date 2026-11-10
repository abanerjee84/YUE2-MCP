"""Run an isolated real CUDA job and save audio integrity and VRAM evidence."""

import argparse
import json
import math
import time
import wave
from dataclasses import replace
from threading import Event

import numpy as np

from yue_music.backend import YueBackend
from yue_music.config import ROOT, gpu_inventory, load_settings
from yue_music.schema import MusicRequest

parser = argparse.ArgumentParser()
parser.add_argument("--quant", choices=["Q8_0", "Q6_K"], default="Q8_0")
parser.add_argument("--duration", type=float, default=30)
parser.add_argument("--cot", choices=["full", "off", "melody"], default="full")
parser.add_argument("--max-seq", type=int, default=16384)
parser.add_argument("--name", default="smoke")
parser.add_argument("--keep-loaded", action="store_true")
parser.add_argument("--repeats", type=int, default=1)
args = parser.parse_args()
settings = replace(
    load_settings(),
    quant=args.quant,
    max_seq=args.max_seq,
    keep_loaded=args.keep_loaded,
)
(settings.home / "outputs").mkdir(exist_ok=True)
backend = YueBackend(settings)
req = MusicRequest(
    style="English, indie folk pop, warm clear lead vocal, acoustic guitar, bass, gentle drums, 100 BPM",
    lyrics="[Verse]\nThe morning train rolls past the hill\nThe river shines, the town is still\nI take the road that leads to you\nThe sky unfolds in gold and blue\n\n[Chorus]\nCarry me home through the light\nWe keep a song through the night\nOne little spark, one open door\nWe find the world we're looking for",
    duration=args.duration,
    cot=args.cot,
    steps=8,
    seed=7,
    lm_seed=7,
)
if args.duration > 90:
    req = req.model_copy(update={"lyrics": "\n\n".join([req.lyrics] * 4)})
job_id = __import__("uuid").uuid4().hex
print(
    json.dumps({"job_id": job_id, "quant": args.quant, "duration": args.duration}),
    flush=True,
)
report = {
    "case": args.name,
    "request": req.model_dump(),
    "hardware": gpu_inventory(),
    "max_seq": settings.max_seq,
    "keep_loaded": settings.keep_loaded,
}
started = time.monotonic()
try:
    result = backend.generate(req, job_id, lambda stage, percent: None, Event())
    with wave.open(result["audio_path"]) as wav:
        audio = (
            np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(
                np.float32
            )
            / 32768
        )
        rms = float(np.sqrt(np.mean(audio**2)))
        report["audio"] = {
            "channels": wav.getnchannels(),
            "sample_rate": wav.getframerate(),
            "duration_seconds": wav.getnframes() / wav.getframerate(),
            "rms": rms,
            "peak": float(np.max(np.abs(audio))),
            "finite": bool(np.isfinite(audio).all()),
        }
        if not math.isfinite(rms) or rms < 1e-5:
            raise RuntimeError("Generated audio is silent or nonfinite")
    report.update(status="succeeded", result=result)
    report["runs"] = [result]
    for _repeat in range(1, args.repeats):
        next_id = __import__("uuid").uuid4().hex
        repeated = backend.generate(req, next_id, lambda stage, percent: None, Event())
        report["runs"].append(repeated)
    print(json.dumps(report, indent=2), flush=True)
except Exception as exc:
    report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    print(json.dumps(report, indent=2), flush=True)
finally:
    backend.unload()
    report["wall_seconds"] = round(time.monotonic() - started, 3)
    path = ROOT / "docs" / f"benchmark-{args.name}.json"
    path.write_text(json.dumps(report, indent=2), "utf-8")
raise SystemExit(0 if report["status"] == "succeeded" else 1)
