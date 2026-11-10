import asyncio
import json

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent, ToolAnnotations

from .client import Client
from .schema import MusicRequest


def structured(value):
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(value))],
        structuredContent=value,
    )


def create_mcp(settings, autostart=True):
    client = Client(settings, autostart=autostart)
    mcp = FastMCP(
        "local-yue",
        host="127.0.0.1",
        port=settings.port,
        streamable_http_path="/mcp",
        stateless_http=True,
        instructions="Local YuE2 GGUF music on RTX 3060. Supply style and lyrics. "
        "Submissions return job_id immediately; poll get_music_job. Duration is a ceiling. "
        "Full mode writes an editable ABC score. Outputs are local audio, score and replay files. "
        "Weights are CC BY-NC 4.0. Batch one; status reports the configured residency profile.",
    )

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False, destructiveHint=False, openWorldHint=False
        )
    )
    async def generate_music(
        style: str,
        lyrics: str = "",
        duration: float = 60,
        cot: str = "full",
        abc: str = "",
        steps: int = 8,
        seed: int | None = None,
        lm_seed: int | None = None,
        output_format: str = "wav16",
        client_request_id: str | None = None,
    ) -> CallToolResult:
        """Queue music from style and section-tagged lyrics; return immediately with job_id.

        duration is a 10-300 second ceiling, not an exact length. Empty lyrics allow instrumental
        generation. cot is full, melody or off. Supply abc in full/melody to render an edited score.
        seed=-1 or omitted selects a random seed. Omitted lm_seed inherits it;
        lm_seed=-1 selects an independent random token seed.
        """
        req = MusicRequest(
            style=style,
            lyrics=lyrics,
            duration=duration,
            cot=cot,
            abc=abc,
            steps=steps,
            seed=seed,
            lm_seed=lm_seed,
            output_format=output_format,
            client_request_id=client_request_id,
        )
        return structured(await asyncio.to_thread(client.submit, req.model_dump()))

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    async def get_music_job(job_id: str) -> CallToolResult:
        """Get status, stage, output paths, duration, seeds and sampled dedicated/shared VRAM."""
        return structured(await asyncio.to_thread(client.job, job_id))

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False, destructiveHint=False, openWorldHint=False
        )
    )
    async def cancel_music_job(job_id: str) -> CallToolResult:
        """Cancel a queued job or stop the native worker serving the current job."""
        return structured(await asyncio.to_thread(client.cancel, job_id))

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    async def music_service_status() -> CallToolResult:
        """Report model availability, GPU UUID, quantization, memory policy and latest VRAM."""
        return structured(await asyncio.to_thread(client.health))

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    async def get_music_score(job_id: str) -> CallToolResult:
        """Return the successful job's ABC score for inspection or musical editing."""
        score = await asyncio.to_thread(client.artifact, job_id, "score")
        return structured({"job_id": job_id, "abc": score.decode("utf-8")})

    return mcp
