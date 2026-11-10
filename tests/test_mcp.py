"""Exercise real subprocess stdio and HTTP MCP sessions against an isolated service."""

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from yue_music.client import Client
from yue_music.config import Settings


@pytest.fixture
def service(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    settings = Settings(
        home=tmp_path, port=port, token="test-only-secret", keep_loaded=False
    )
    settings.save()
    env = os.environ.copy()
    env["YUE_MUSIC_HOME"] = str(tmp_path)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    log = (tmp_path / "server.log").open("wb")
    process = subprocess.Popen(
        [sys.executable, "-m", "yue_music", "serve"],
        env=env,
        stdout=log,
        stderr=log,
        stdin=subprocess.DEVNULL,
    )
    client = Client(settings, autostart=False)
    try:
        deadline = time.monotonic() + 15
        while not client.reachable():
            if process.poll() is not None or time.monotonic() > deadline:
                raise AssertionError((tmp_path / "server.log").read_text("utf-8"))
            time.sleep(0.1)
        yield settings, env
    finally:
        process.terminate()
        process.wait(timeout=15)
        log.close()


async def exercise(session):
    await session.initialize()
    names = {t.name for t in (await session.list_tools()).tools}
    assert names == {
        "generate_music",
        "get_music_job",
        "cancel_music_job",
        "music_service_status",
        "get_music_score",
    }
    result = await session.call_tool("music_service_status", {})
    assert not result.isError
    text = "\n".join(c.text for c in result.content if c.type == "text")
    assert '"weight_policy": "strict_stage_eviction"' in text
    assert '"batch_size": 1' in text
    # These tests intentionally have no model weights. A real submission must fail honestly.
    submitted = await session.call_tool(
        "generate_music", {"style": "English, indie pop", "duration": 10}
    )
    assert not submitted.isError
    job_id = submitted.structuredContent["job_id"]
    for _ in range(40):
        response = await session.call_tool("get_music_job", {"job_id": job_id})
        if response.structuredContent["status"] == "failed":
            assert "setup-models" in response.structuredContent["error"]
            break
        await asyncio.sleep(0.1)
    else:
        raise AssertionError("Missing models did not produce an actionable error")


async def test_stdio_mcp(service):
    settings, env = service
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "yue_music", "mcp"], env=env
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(
            read, write, read_timeout_seconds=timedelta(seconds=15)
        ) as session:
            await exercise(session)


async def test_http_mcp(service):
    settings, env = service
    async with httpx.AsyncClient(
        headers={"Authorization": "Bearer " + settings.token}, trust_env=False
    ) as http:
        async with streamable_http_client(settings.url + "/mcp", http_client=http) as (
            read,
            write,
            _,
        ):
            async with ClientSession(
                read, write, read_timeout_seconds=timedelta(seconds=15)
            ) as session:
                await exercise(session)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows driver environment")
def test_filtered_mcp_environment_keeps_nvml_available():
    from mcp.client.stdio import get_default_environment

    from yue_music.config import gpu_inventory

    baseline = gpu_inventory()
    if not baseline:
        pytest.skip("No local NVIDIA inventory")
    code = "import json; from yue_music.config import gpu_inventory; print(json.dumps(gpu_inventory()))"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=get_default_environment(),
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    assert {g["uuid"] for g in json.loads(result.stdout)} == {
        g["uuid"] for g in baseline
    }


@pytest.mark.skipif(sys.platform != "win32", reason="Windows MCP process-job lifetime")
async def test_autostart_survives_adapter_disconnect(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    settings = Settings(home=tmp_path, port=port, token="test-lifetime-token")
    settings.save()
    client = Client(settings, autostart=False)
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "yue_music", "mcp"],
        env={"YUE_MUSIC_HOME": str(tmp_path)},
    )
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("music_service_status", {})
                assert not result.isError
        assert client.reachable(), "Shared worker died with its stdio adapter"
    finally:
        if client.reachable():
            client._request("POST", "/admin/shutdown")
        deadline = time.monotonic() + 10
        while client.reachable() and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
