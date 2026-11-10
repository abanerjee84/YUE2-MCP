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


@pytest.mark.parametrize(
    "seed,lm_seed,draws,expected",
    [
        (None, None, [17], (17, 17)),
        (-1, None, [17], (17, 17)),
        (-1, -1, [17, 29], (17, 29)),
        (0, -1, [29], (0, 29)),
        (2**32 - 1, None, [], (2**32 - 1, 2**32 - 1)),
        (-1, 42, [17], (17, 42)),
        (7, 8, [], (7, 8)),
    ],
)
def test_seed_resolution_is_persisted_and_retries_do_not_redraw(
    tmp_path, monkeypatch, seed, lm_seed, draws, expected
):
    from unittest.mock import Mock

    draw = Mock(side_effect=draws)
    monkeypatch.setattr("yue_music.jobs.secrets.randbelow", draw)
    store = JobStore(tmp_path, StubBackend(tmp_path))
    req = MusicRequest(
        style="pop", seed=seed, lm_seed=lm_seed, client_request_id="seed-retry"
    )
    first = store.submit(req)
    assert (first["request"]["seed"], first["request"]["lm_seed"]) == expected
    assert (req.seed, req.lm_seed) == (seed, lm_seed)
    assert store.submit(req)["job_id"] == first["job_id"]
    recovered = JobStore(tmp_path, StubBackend(tmp_path))
    retried = recovered.submit(req)
    assert (retried["request"]["seed"], retried["request"]["lm_seed"]) == expected
    assert draw.call_count == len(draws)


def test_random_seed_is_drawn_for_each_new_job(tmp_path, monkeypatch):
    from unittest.mock import Mock

    monkeypatch.setattr("yue_music.jobs.secrets.randbelow", Mock(side_effect=[17, 29]))
    store = JobStore(tmp_path, StubBackend(tmp_path))
    req = MusicRequest(style="pop", seed=-1)
    first, second = store.submit(req), store.submit(req)
    assert (first["request"]["seed"], first["request"]["lm_seed"]) == (17, 17)
    assert (second["request"]["seed"], second["request"]["lm_seed"]) == (29, 29)


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
        job = http.post("/v1/jobs", json={"style": "pop", "seed": -1, "lm_seed": -1})
        assert job.status_code == 202
        assert 0 <= job.json()["request"]["seed"] < 2**32
        assert 0 <= job.json()["request"]["lm_seed"] < 2**32
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
        {"seed": -2},
        {"lm_seed": -2},
        {"seed": 2**32},
        {"lm_seed": 2**32},
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
