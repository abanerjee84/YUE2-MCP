"""Exercise generation, score retrieval and cancellation through real stdio MCP."""

import asyncio
import json
import sys
from datetime import timedelta

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from yue_music.config import ROOT


async def main():
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "yue_music", "mcp"]
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(
            read, write, read_timeout_seconds=timedelta(seconds=60)
        ) as session:
            await session.initialize()
            submitted = await session.call_tool(
                "generate_music",
                {
                    "style": "English, acoustic folk, clear warm vocal, fingerpicked guitar, gentle percussion",
                    "lyrics": "[Verse]\nThe river carries morning light\nWe leave the shadows of the night\n[Chorus]\nCarry me home, carry me home\nUnder the sky we call our own",
                    "duration": 20,
                    "seed": 17,
                    "steps": 8,
                    "cot": "full",
                },
            )
            assert not submitted.isError, submitted
            job_id = submitted.structuredContent["job_id"]
            for _ in range(2400):
                reply = await session.call_tool("get_music_job", {"job_id": job_id})
                job = reply.structuredContent
                if job["status"] in {"succeeded", "failed", "cancelled"}:
                    break
                await asyncio.sleep(0.5)
            assert job["status"] == "succeeded", job
            score = await session.call_tool("get_music_score", {"job_id": job_id})
            assert not score.isError and score.structuredContent["abc"].strip()
            cancel_submission = await session.call_tool(
                "generate_music",
                {
                    "style": "English, folk pop",
                    "lyrics": "[Verse]\nThe morning rises over town",
                    "duration": 60,
                },
            )
            cancelled_id = cancel_submission.structuredContent["job_id"]
            await asyncio.sleep(2)
            await session.call_tool("cancel_music_job", {"job_id": cancelled_id})
            for _ in range(60):
                cancelled = (
                    await session.call_tool("get_music_job", {"job_id": cancelled_id})
                ).structuredContent
                if cancelled["status"] == "cancelled":
                    break
                await asyncio.sleep(0.5)
            assert cancelled["status"] == "cancelled", cancelled
            report = {
                "transport": "stdio MCP",
                "generation": job,
                "score_chars": len(score.structuredContent["abc"]),
                "cancellation_status": cancelled["status"],
            }
            (ROOT / "docs/live-mcp-validation.json").write_text(
                json.dumps(report, indent=2), "utf-8"
            )
            print(json.dumps(report, indent=2))


asyncio.run(main())
