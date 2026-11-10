import json
import threading
import time
import wave

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from yue_music.backend import parse_result
from yue_music.config import Settings
from yue_music.jobs import JobStore
from yue_music.schema import MusicRequest
from yue_music.service import create_app


class StubBackend:
    def __init__(self, home):
        self.home, self.active, self.peak = home, 0, 0
        self.lock = threading.Lock()

    def readiness(self):
        return {"ready": True}

    def unload(self):
        pass

    def generate(self, req, job_id, progress, cancelled):
        with self.lock:
            self.active += 1
            self.peak = max(self.active, self.peak)
        try:
            for i in range(12):
                progress("test", i)
                time.sleep(0.005)
            path = self.home / "outputs" / (job_id + ".wav")
            with wave.open(str(path), "wb") as wav:
                wav.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
                wav.writeframes(bytes(1920))
            path.with_suffix(".abc").write_text("X:1\nK:C\nC D E F|", "utf-8")
            path.with_suffix(".replay.json").write_text("{}")
            return {"audio_path": str(path)}
        finally:
            self.active -= 1


def wait(store, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = store.get(job_id)
        if result["status"] in {"succeeded", "failed", "cancelled"}:
            return result
        time.sleep(0.005)
    raise AssertionError("Job did not finish")


def test_serialization_retry_and_both_seeds(tmp_path):
    backend = StubBackend(tmp_path)
    store = JobStore(tmp_path, backend)
    req = MusicRequest(style="indie pop", client_request_id="same")
    first = store.submit(req)
    assert first["request"]["seed"] == first["request"]["lm_seed"]
    assert store.submit(req)["job_id"] == first["job_id"]
    with pytest.raises(ValueError, match="different request"):
        store.submit(MusicRequest(style="jazz", client_request_id="same"))
    jobs = [first] + [store.submit(MusicRequest(style="pop")) for _ in range(3)]
    store.start()
    try:
        for job in jobs:
            assert wait(store, job["job_id"])["status"] == "succeeded"
        assert backend.peak == 1
    finally:
        store.close()


def test_cancel_and_restart(tmp_path):
    store = JobStore(tmp_path, StubBackend(tmp_path))
    job = store.submit(MusicRequest(style="pop"))
    assert store.cancel(job["job_id"])["status"] == "cancelled"
    running = store.submit(MusicRequest(style="jazz"))
    store.update(running["job_id"], status="running")
    reopened = JobStore(tmp_path, StubBackend(tmp_path))
    assert reopened.get(running["job_id"])["status"] == "failed"


def test_http_auth_validation_and_artifacts(tmp_path):
    settings = Settings(home=tmp_path, token="test-token")
    with TestClient(create_app(settings, StubBackend(tmp_path))) as http:
        assert http.get("/health").status_code == 401
        http.headers["Authorization"] = "Bearer test-token"
        assert (
            http.get("/health", headers={"Origin": "https://other.example"}).status_code
            == 403
        )
        assert (
            http.post("/v1/jobs", json={"style": "pop", "duration": 999}).status_code
            == 422
        )
        assert (
            http.post(
                "/v1/jobs", json={"style": "pop", "path": "arbitrary"}
            ).status_code
            == 422
        )
        job = http.post("/v1/jobs", json={"style": "pop"})
        assert job.status_code == 202
        job_id = job.json()["job_id"]
        result = wait(http.app.state.store, job_id)
        assert result["status"] == "succeeded"
        assert http.get(f"/v1/jobs/{job_id}/audio").content.startswith(b"RIFF")
        assert "K:C" in http.get(f"/v1/jobs/{job_id}/score").text
        assert http.get(f"/v1/jobs/{job_id}/config.json").status_code == 404


def test_request_bounds():
    for data in (
        {"duration": float("nan")},
        {"steps": 8.5},
        {"seed": -1},
        {"abc": "K:C", "cot": "off"},
    ):
        with pytest.raises(ValidationError):
            MusicRequest(style="pop", **data)


def test_native_multipart_binary_integrity():
    audio = b"RIFF\x00\xff\r\n\x00binary"
    metadata = json.dumps({"abc": "X:1\nK:C", "semantic_tokens": "1,2"}).encode()
    body = (
        b"--result\r\nContent-Type: application/json\r\n\r\n"
        + metadata
        + b"\r\n--result\r\nContent-Type: audio/wav\r\n\r\n"
        + audio
        + b"\r\n--result--\r\n"
    )
    replay, decoded, _ = parse_result('multipart/mixed; boundary="result"', body)
    assert decoded == audio
    assert replay["abc"].startswith("X:1")
    with pytest.raises(RuntimeError, match="replay metadata"):
        parse_result("multipart/mixed; boundary=result", b"--result--\r\n")
