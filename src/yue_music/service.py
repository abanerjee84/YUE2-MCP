from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .backend import YueBackend
from .config import Settings
from .jobs import JobStore
from .schema import MusicRequest


class LocalAuthMiddleware:
    def __init__(self, app, settings):
        self.app, self.settings = app, settings

    async def __call__(self, scope, receive, send):
        import hmac

        from starlette.datastructures import Headers
        from starlette.responses import JSONResponse

        if scope["type"] == "http":
            headers = Headers(scope=scope)
            if not hmac.compare_digest(
                headers.get("authorization", "").encode(),
                ("Bearer " + self.settings.token).encode(),
            ):
                return await JSONResponse({"detail": "Bearer token required"}, 401)(
                    scope, receive, send
                )
            origin = headers.get("origin")
            if origin and origin not in {
                self.settings.url,
                f"http://localhost:{self.settings.port}",
            }:
                return await JSONResponse({"detail": "Origin rejected"}, 403)(
                    scope, receive, send
                )
        return await self.app(scope, receive, send)


def create_app(settings: Settings, backend=None):
    from .mcp_adapter import create_mcp

    backend = backend or YueBackend(settings)
    store = JobStore(settings.home, backend)
    mcp = create_mcp(settings, autostart=False)

    @asynccontextmanager
    async def lifespan(app):
        store.start()
        try:
            async with mcp.session_manager.run():
                yield
        finally:
            store.close()
            backend.unload()

    app = FastAPI(title="YuE2 Music Local", version="0.1.0", lifespan=lifespan)
    app.state.store = store
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"]
    )
    app.add_middleware(LocalAuthMiddleware, settings=settings)

    @app.get("/health")
    def health():
        return {
            "service": "yue-music-local",
            "version": "0.1.0",
            "home": str(settings.home),
            "inference": backend.readiness(),
        }

    @app.post("/admin/shutdown")
    def shutdown(tasks: BackgroundTasks):
        callback = getattr(app.state, "shutdown", None)
        if callback is None:
            raise HTTPException(409, "Service is externally managed")
        tasks.add_task(callback)
        return {"status": "stopping"}

    @app.post("/v1/jobs", status_code=202)
    def submit(req: MusicRequest):
        try:
            return store.submit(req)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        except OverflowError as exc:
            raise HTTPException(429, str(exc)) from exc

    @app.get("/v1/jobs/{job_id}")
    def status(job_id: str):
        try:
            return store.get(job_id)
        except KeyError:
            raise HTTPException(404, "Unknown job") from None

    @app.post("/v1/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        try:
            return store.cancel(job_id)
        except KeyError:
            raise HTTPException(404, "Unknown job") from None

    @app.get("/v1/jobs/{job_id}/{artifact}")
    def output(job_id: str, artifact: str):
        job = status(job_id)
        if job["status"] != "succeeded":
            raise HTTPException(409, "Output is not ready")
        if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
            raise HTTPException(404)
        ext = ".mp3" if job["request"]["output_format"] == "mp3" else ".wav"
        suffixes = {
            "audio": (ext, "audio/mpeg" if ext == ".mp3" else "audio/wav"),
            "score": (".abc", "text/plain"),
            "replay": (".replay.json", "application/json"),
        }
        if artifact not in suffixes:
            raise HTTPException(404, "Unknown artifact")
        suffix, mime = suffixes[artifact]
        return FileResponse(
            settings.home / "outputs" / (job_id + suffix), media_type=mime
        )

    app.mount("/", mcp.streamable_http_app())
    return app
