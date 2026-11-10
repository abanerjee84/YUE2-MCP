from __future__ import annotations

import os
import subprocess
import sys
import time

import httpx
from filelock import FileLock

from .config import Settings, runtime_environment
from .jobs import TERMINAL


class Client:
    def __init__(self, settings: Settings, autostart=True):
        self.settings = settings
        self.autostart = autostart

    def _request(self, method, path, **kwargs):
        with httpx.Client(
            base_url=self.settings.url,
            timeout=30,
            trust_env=False,
            headers={"Authorization": "Bearer " + self.settings.token},
        ) as http:
            response = http.request(method, path, **kwargs)
        if response.is_error:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1500]}")
        return response

    def reachable(self):
        try:
            data = self._request("GET", "/health").json()
            return data.get("service") == "yue-music-local" and data.get("home") == str(
                self.settings.home
            )
        except (httpx.HTTPError, RuntimeError):
            return False

    def ensure(self):
        if self.reachable():
            return
        if not self.autostart:
            raise RuntimeError("Music service is unavailable")
        with FileLock(str(self.settings.home / "startup.lock"), timeout=25):
            if self.reachable():
                return
            env = runtime_environment()
            env["YUE_MUSIC_HOME"] = str(self.settings.home)
            process = None
            if os.name == "nt":
                from .windows import start_service

                start_service(
                    [sys.executable, "-m", "yue_music", "serve", "--background"],
                    env,
                    self.settings.home,
                )
            else:
                with (self.settings.home / "service.log").open("ab") as log:
                    process = subprocess.Popen(
                        [sys.executable, "-m", "yue_music", "serve"],
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=log,
                        start_new_session=True,
                    )
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if self.reachable():
                    return
                if process is not None and process.poll() is not None:
                    break
                time.sleep(0.2)
            raise RuntimeError(
                f"Service did not start. See {self.settings.home / 'service.log'}"
            )

    def health(self):
        self.ensure()
        return self._request("GET", "/health").json()

    def submit(self, data: dict):
        self.ensure()
        return self._request("POST", "/v1/jobs", json=data).json()

    def job(self, job_id: str):
        self.ensure()
        return self._request("GET", f"/v1/jobs/{job_id}").json()

    def cancel(self, job_id: str):
        self.ensure()
        return self._request("POST", f"/v1/jobs/{job_id}/cancel").json()

    def artifact(self, job_id: str, artifact: str = "audio"):
        self.ensure()
        return self._request("GET", f"/v1/jobs/{job_id}/{artifact}").content

    def wait(self, job_id: str, timeout: float = 1800):
        deadline = time.monotonic() + timeout
        while True:
            result = self.job(job_id)
            if result["status"] in TERMINAL:
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Still running; resume with yue-music status {job_id}"
                )
            time.sleep(0.5)
